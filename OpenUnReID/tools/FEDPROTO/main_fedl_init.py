from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
import tensorflow as tf
from sklearn import manifold
import tensorboard as tb
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import PIL.Image
from openunreid.models.utils.extract import extract_features, extract_features_for_similarities, extract_features_protos
from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.metrics.accuracy import accuracy
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader, build_val_dataloader
from openunreid.data.builder import build_train_dataloader_for_sim

from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.utils.config import (
    cfg,
    cfg_s,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.torch_utils import copy_state_dict, load_checkpoint
from openunreid.utils.dist_utils import init_dist, synchronize, get_dist_info
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.apis.test import val_reid
import copy
import io
import collections
from torch.utils.data import TensorDataset, DataLoader
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
def add_model(model_1, model_2, w_1, w_2):
    if model_1 == None:
        params2 = model_2.named_parameters()
        dict_params2 = dict(params2)
        with torch.no_grad():
            for name2 in dict_params2:
                dict_params2[name2].set_(dict_params2[name2].data*w_2)
        return model_2
    params1 = model_1.named_parameters()
    params2 = model_2.named_parameters()
    dict_params2 = dict(params2)
    with torch.no_grad():
        for name1, param1 in params1:
            if name1 in dict_params2 and ("clas" not in name1):
                dict_params2[name1].set_(param1.data*w_1 + dict_params2[name1].data*w_2)
    return model_2

def aggregate_models (models,weights_fed_avg=None,beta=0.2):
    alpha = len(models)
    model = add_model(None, models[0], 0, beta) #1/alpha*
    for i in range(1,len(models)):
        if weights_fed_avg is None:
            model = add_model(model, models[i], 1, ((1-beta)/(alpha-1))) #((1-beta)/(alpha-1))
        else:
            model = add_model(model, models[i], 1, ((1-beta)*(weights_fed_avg[i-1])))
    return model
def gen_plot(Y,color):
    """Create a pyplot plot and save to buffer."""
    plt.figure(figsize=(15,8))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
def filtre_source_(train_sets,features,tag,writer,rank):
    target = train_sets[0]
    source = train_sets[1]
    cos = torch.nn.CosineSimilarity(dim=-1)
    features_target = features["target"]
    features_source = features["source"]
    id_to_indice_t = {id:indice for id,indice in enumerate(features_target.keys())}
    id_to_indice_s = {id:indice for id,indice in enumerate(features_source.keys())}
    similarities = torch.zeros(len(id_to_indice_t),len(id_to_indice_s))
    for i in range(len(id_to_indice_t.keys())): 
        for j in range(len(features_source)):
            similarities[i,j]= cos(features_target[id_to_indice_t[list(id_to_indice_t.keys())[i]]],features_source[id_to_indice_s[j]])
    indices_min_source = similarities.max(axis=1)[1]
    similarities_min_source = similarities.max(axis=1)[0]
    fst_quantile = torch.quantile(torch.tensor(similarities.max(axis=1)[0]),0.25)
    trd_quantile = torch.quantile(torch.tensor(similarities.max(axis=1)[0]),0.75)
    target_similar_source = {}
    target_images = []
    source_images = []
    images_to_show_fstq = []
    images_to_show_trdq = []
    source_image_ids = []
    source_image_paths = []
    for i in range(len(indices_min_source)):
        target_similar_source[id_to_indice_t[list(id_to_indice_t.keys())[i]]]=id_to_indice_s[indices_min_source[i].item()]
        try:
            target_image = target._get_single_item(id_to_indice_t[list(id_to_indice_t.keys())[i]],for_sim=True)["img"]
            source_image = source._get_single_item(id_to_indice_s[indices_min_source[i].item()],for_sim=True)["img"]
            source_image_id = source._get_single_item(id_to_indice_s[indices_min_source[i].item()],for_sim=True)["id"]
            source_image_path = source._get_single_item(id_to_indice_s[indices_min_source[i].item()],for_sim=True)["path"]
            if similarities_min_source[i]<=fst_quantile:
                image_to_show_fstq = torchvision.utils.make_grid([target_image,source_image])
                images_to_show_fstq.append(image_to_show_fstq)

            elif similarities_min_source[i]>=trd_quantile:
                image_to_show_trdq = torchvision.utils.make_grid([target_image,source_image])
                images_to_show_trdq.append(image_to_show_trdq)
            if source_image_id not in source_image_ids:
                source_image_ids.append(source_image_id)
            source_image_paths.append(source_image_path)
            
        except:
            continue
    images_to_show_trdq = torchvision.utils.make_grid(images_to_show_trdq)
    images_to_show_fstq = torchvision.utils.make_grid(images_to_show_fstq)
    if rank == 0:
        writer.add_image(tag + ' images target / source 1st q', images_to_show_fstq, 0)
        writer.add_image(tag + ' images target / source 3rd q', images_to_show_trdq, 0)
        writer.add_histogram(tag + ' similarities target / source', similarities.max(axis=1)[0], 0)
        print("done images / similarities")
    return similarities,target_similar_source,source_image_ids,source_image_paths


# n_tasks = 6 #6
class MMTRunner(BaseRunner):
    def train_step(self, iter, batch, batch_protos=None, batch_target=None):
        task_id = self.cfg.task_id
        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        len_ = int(len(data["id"])/2)
        
        
        inputs_1 = data["img"][0].cuda()
        targets = data["id"].cuda()
        inputs_mo1 = inputs_1 #torch.cat((inputs_1,inputs_similarities_1))
        
        results_1, results_1_mean = self.model(inputs_mo1)
        results_target_features = results_1["feat"] #for mmd optimization

        results_1["prob"] = results_1["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]
        
        results_1_mean["prob"] = results_1_mean["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]

        for key in results_1.keys():
            results_1[key]=results_1[key][:len(inputs_1)]
        for key in results_1_mean.keys():
            results_1_mean[key]=results_1_mean[key][:len(inputs_1)]
        total_loss = 0
        meters = {}
        for key in self.criterions.keys():
            if key == "soft_entropy":
                loss = self.criterions[key](
                    results_1, results_1_mean
                )[0] #+ self.criterions[key](results_2, results_1_mean)[0]
                loss_vec = self.criterions[key](
                    results_1, results_1_mean
                )[1] #+ self.criterions[key](results_2, results_1_mean)[1]
                #print("loss vec {}".format(loss_vec))
            elif key == "soft_softmax_triplet":
                loss = self.criterions[key](
                        results_1, targets, results_1_mean
                ) #+ self.criterions[key](results_2, targets, results_1_mean)
            elif key == "cross_entropy":
                loss = self.criterions[key](results_1, targets)[0] #+ self.criterions[key](results_2, targets)[0]
            elif key == "DCL":
                if self.model_server is not None:
                    loss =  sum(((x - y).abs()**2).sum() for x, y in zip(self.model.module.net.backbone.state_dict().values(), self.model_server.state_dict().values())).sqrt()
                else:
                    loss = loss*0
            elif key == "MMD_loss": 
                if batch_protos is not None: 
                    loss = self.criterions[key](results_target_features,batch_protos[0]) #+ self.criterions[key](results_2_source,results_2_target) 
                else:
                    loss = loss*0
            else:
                loss = self.criterions[key](results_1, targets)
            if iter==self.cfg.TRAIN.iters-1:
                if key=="soft_entropy" or key=="cross_entropy":
                    if self._rank == 0:
                        print("in rank 0 loss")
                        self.writer.add_scalar('losses_'+key+"/"+str(task_id),loss,self._epoch)
                else:  
                    if self._rank == 0:  
                        print("in rank 0 loss")
                        self.writer.add_scalar('losses_'+key+"/"+str(task_id),loss,self._epoch)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])   
            meters[key] = loss.item()
        
        self.losses.append(total_loss)
       
        acc_1 = accuracy(results_1["prob"].data, targets.data)
        meters["Acc@1"] = acc_1[0] 
        self.accs.append(meters["Acc@1"])
        if (iter==self.cfg.TRAIN.iters-1) & (self._rank==0):
            t_loss = sum(self.losses)/len(self.losses)
            acc_ = sum(self.accs)/len(self.accs)
            self.writer.add_scalar("TRAIN/Acc/"+str(task_id),acc_,self._epoch)
            self.writer.add_scalar("TRAIN/Total_loss/"+str(task_id),t_loss,self._epoch)
            self.losses = []
            self.accs = []
        self.train_progress.update(meters)
        return total_loss

def parge_config():
    parser = argparse.ArgumentParser(description="FED-SO-PROTO training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument("--MMDloss", type=float, default=0.)
    parser.add_argument("--epochs_s", type=int, default=0)
    parser.add_argument("--iters_s", type=int, default=0)
    parser.add_argument("--epochs_t", type=int, default=0)
    parser.add_argument("--iters_t", type=int, default=0)
    parser.add_argument("--times_itc", type=int, default=0)

    parser.add_argument("--exec", type=float) #if 0 then local, 1 fastml, and 2 Telecom Cluster
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()
    cfg_from_yaml_file(args.config, cfg)
    try:
        # cfg_from_yaml_file("./fedl/data/OpenUnReID/tools/FEDPROTO/config_s.yaml", cfg_s)
        cfg_from_yaml_file("./cvlab-federated-phd/OpenUnReID/tools/FEDPROTO/config_s.yaml", cfg_s)
    except:
        try: 
            cfg_from_yaml_file("./fedl/data/OpenUnReID/tools/FEDPROTO/config_s.yaml", cfg_s)
        except:
            cfg_from_yaml_file("/exp/OpenUnReID/tools/FEDPROTO/config_s.yaml", cfg_s)
    # assert cfg.TRAIN.PSEUDO_LABELS.use_outliers
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg_s.launcher = args.launcher
    cfg_s.tcp_port = args.tcp_port

    if args.epochs_s>0:
        cfg_s.TRAIN.epochs = args.epochs_s #else refer to config_s.yaml
    if args.epochs_t>0:
        cfg.TRAIN.epochs = args.epochs_t #else refer to config.yaml
    if args.iters_s>0:
        cfg_s.TRAIN.iters = args.iters_s
    else:
        cfg_s.TRAIN.iters = 0 # iters = Ic/num_id_per_batch personalized per client
    if args.iters_t>0:
        cfg.TRAIN.iters = args.iters_t
    else:
        cfg.TRAIN.iters = 0 # iters = Ic/num_id_per_batch personalized per client
    cfg.TRAIN.times_itc = args.times_itc
    if args.MMDloss>0:
        cfg.TRAIN.LOSS.losses["MMD_loss"] = args.MMDloss #else refer to config.yaml
    cfg.exec = args.exec
    cfg_s.exec = args.exec
    if cfg.exec==0:
        print("Training locally on naboo")
        cfg.DATA_ROOT = Path(cfg.DATA_ROOT_local)
        cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT_local)
        cfg.MODEL.backbone_path = cfg.MODEL.backbone_path_local
        cfg.MODEL.source_pretrained = cfg.MODEL.source_pretrained_local

        cfg_s.DATA_ROOT = Path(cfg_s.DATA_ROOT_local)
        cfg_s.LOGS_ROOT = Path(cfg_s.LOGS_ROOT_local)
        cfg_s.MODEL.backbone_path = cfg_s.MODEL.backbone_path_local
        cfg_s.MODEL.source_pretrained = cfg_s.MODEL.source_pretrained_local
    elif cfg.exec==2:
        print("training on Telecom Cluster")
        cfg.DATA_ROOT = Path(cfg.DATA_ROOT_cluster)
        cfg.LOGS_ROOT = Path(cfg.LOGS_ROOT_cluster)
        cfg.MODEL.backbone_path = cfg.MODEL.backbone_path_cluster
        cfg.MODEL.source_pretrained = cfg.MODEL.source_pretrained_cluster

        cfg_s.DATA_ROOT = Path(cfg_s.DATA_ROOT_cluster)
        cfg_s.LOGS_ROOT = Path(cfg_s.LOGS_ROOT_cluster)
        cfg_s.MODEL.backbone_path = cfg_s.MODEL.backbone_path_cluster
        cfg_s.MODEL.source_pretrained = cfg_s.MODEL.source_pretrained_cluster

    else:
        print("training on fastml")
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    cfg_s.work_dir = cfg.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()
    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    rank,_,_ = get_dist_info()
    print(rank)
    if rank == 0:
        if cfg.exec==0:
            writer = SummaryWriter(cfg.work_dir / 'logs_tb/',flush_secs=10)
        elif cfg.exec==2:
            writer = SummaryWriter(cfg.work_dir / 'logs_tb/',flush_secs=10)
        else:
            writer = SummaryWriter('/out/logs_tb/'+str(cfg.TRAIN.LOSS.losses["MMD_loss"])+'/',flush_secs=10)
    else :
        writer = None
    
    synchronize()
    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # init federated variables
    n_tasks_dict = cfg.FED.num_cameras
    dataset_target = list(cfg.TRAIN.datasets.keys())[0]
    n_tasks = n_tasks_dict[dataset_target]
    client_sizes = cfg.FED.client_sizes[dataset_target]
    cfg.n_tasks = n_tasks
    cfg_s.n_tasks = n_tasks
    cfg.task_id=-1
    cfg_s.task_id=-1
    # build train loader
    train_loader_s, train_sets_s = build_train_dataloader(cfg_s, n_tasks,
                0,[False], joint=False)
    train_loaders = [train_loader_s[0]]
    train_setss = [train_sets_s]
    for i in range (n_tasks):
        train_loader, train_sets = build_train_dataloader(cfg, n_tasks,
                i,[True,False], joint=False)
        train_loaders.append([train_loader[0]])
        train_setss.append([train_sets[0]])

    # train_loader_target, train_sets_target = build_train_dataloader(cfg, n_tasks, 0 ,[False], joint=False)
    # the number of classes for the model is tricky,
    # you need to make sure that
    # it is always larger than the number of clusters
    num_classes_ = []
    for i in range(n_tasks+1):
        if i == 0:
            num_classes_.append(train_setss[0][0].num_pids)
            pass
        num_classes = 0
        for idx, set in enumerate(train_setss[i]):
            if idx in cfg.TRAIN.unsup_dataset_indexes:
                # number of clusters in an unsupervised dataset
                # must not be larger than the number of images
                num_classes += len(set)
            else:
                # ground-truth classes for supervised dataset
                num_classes += set.num_pids
        num_classes_.append(num_classes)
    models_1 = []
    for j in range (len(num_classes_)):
        if j == 0:
            # build model no.1
            model_1 = build_model(cfg_s, num_classes_[j],[], init=cfg.MODEL.source_pretrained)
            model_1.cuda()
        else:
            # build model no.1
            model_1 = build_model(cfg, num_classes_[j],[], init=cfg.MODEL.source_pretrained)
            model_1.cuda()
        
        if dist:
            ddp_cfg = {
                "device_ids": [cfg.gpu],
                "output_device": cfg.gpu,
                "find_unused_parameters": True,
            }
            model_1 = torch.nn.parallel.DistributedDataParallel(model_1, **ddp_cfg)
        elif cfg.total_gpus > 1:
            model_1 = torch.nn.DataParallel(model_1)
        
        models_1.append(model_1)
    
    # build optimizer
    optimizers = [build_optimizer([models_1[0],], **cfg_s.TRAIN.OPTIM) ]
    for i in range (n_tasks):
        optimizers.append(build_optimizer([models_1[i+1],], **cfg.TRAIN.OPTIM))
    # build lr_scheduler
    lr_schedulers = []
    for optimizer in optimizers:
        if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
            lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
        else:
            lr_scheduler = None
        lr_schedulers.append(lr_scheduler)
    # build loss functions
    criterions_ = []
    for n, num_classes in enumerate(num_classes_):
        if n == 0:
            criterions = build_loss(cfg_s.TRAIN.LOSS, num_classes=num_classes, cuda=True)
            criterions_.append(criterions)
        else:
            criterions = build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True)
            criterions_.append(criterions)
    # Do we need to start optmizing MMD since 1st round?
    protos_dataloader = None
    if cfg.FED.mmd_start_round == True:
        # # init Protos
        print("Init Protos")
        if dist:
            batch_size = cfg.TRAIN.LOADER.samples_per_gpu
        else:
            batch_size = cfg.TRAIN.LOADER.samples_per_gpu * cfg.total_gpus
        loaders, datasets = build_val_dataloader(
            cfg_s, for_clustering=True, all_datasets=True, SpCL=True, Protos=True
        )
        memory_features = []
        memory_poolings = []
        for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
            features, poolings = extract_features_protos(
                models_1[0], loader, dataset, with_path=False, prefix="Extract: ",
            )
            assert features.size(0) == len(dataset)
            # if idx in cfg.TRAIN.unsup_dataset_indexes:
            #     # init memory for unlabeled data with instance features
            #     memory_features.append(features)
            #     memory_features_target.append(features)
            # else:
                # init memory for labeled data with class centers
            centers_dict = collections.defaultdict(list)
            pooling_dict = collections.defaultdict(list)
            for i, (_, pid, _) in enumerate(dataset):
                centers_dict[pid].append(features[i].unsqueeze(0))
                # pooling_dict[pid].append(poolings[i].unsqueeze(0))
            centers = [
                torch.cat(centers_dict[pid], 0).mean(0)
                for pid in sorted(centers_dict.keys())
            ]
            # centers_pooling = [
            #     torch.cat(pooling_dict[pid], 0).mean(0)
            #     for pid in sorted(pooling_dict.keys())
            # ]
            memory_features.append(torch.stack(centers, 0))
            # memory_poolings.append(torch.stack(centers_pooling, 0))
            # memory_features_source.append(torch.stack(centers, 0))
        del loaders, datasets
        cuda0 = torch.device('cuda:0')
        memory_features = torch.cat(memory_features).cuda() #to(cuda0)
        # memory_poolings = torch.cat(memory_poolings).cuda() #to(cuda0)
        protos_dataset = TensorDataset(memory_features) # create your datset
        protos_dataloader = DataLoader(protos_dataset,batch_size=batch_size, drop_last=True) 

    # build runner
    print("start training at round : 1")
    runners = []
    clients_sizes = []
    for i in range(n_tasks+1):
        if i==0:
            runner = MMTRunner(
            cfg_s,
            models_1[i],
            optimizers[i],
            criterions_[i],
            train_loaders[i],
            writer,
            train_sets=train_setss[i],
            # train_loader_target=train_loader_target[0],
            lr_scheduler=lr_schedulers[i],
            reset_optim=True,
        )
        else:
            runner = MMTRunner(
            cfg,
            models_1[i],
            optimizers[i],
            criterions_[i],
            train_loaders[i],
            writer,
            train_sets=train_setss[i],
            # train_loader_target=train_loader_target[0],
            lr_scheduler=lr_schedulers[i],
            reset_optim=True,
            protos_dataloader=protos_dataloader,
        )
        try:
            runner.run(cam_id=i-1)
            if args.iters_s>0:
                cfg_s.TRAIN.iters = args.iters_s
            else:
                cfg_s.TRAIN.iters = 0 # iters = Ic/num_id_per_batch personalized per client
            if args.iters_t>0:
                cfg.TRAIN.iters = args.iters_t
            else:
                cfg.TRAIN.iters = 0 # iters = Ic/num_id_per_batch personalized per client
            runners.append(runner)
            if i>0:
                clients_sizes.append(client_sizes[i-1])
        except StopIteration as e:
            print(e)
        cfg.task_id+=1
    test_loaders, queries, galleries = build_test_dataloader(cfg, n_tasks, 0, False)
    l= ["source","target"]
    # for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
    #     for r in range (len(runners)):
    #         runner = runners[r]
    #         for idx in range(1):
    #             if cfg.TEST.datasets[i]=="msmt17" and i==0:
    #                 mAP = 0
    #             else:

    #                 print("==> Test on the no.{} model".format(idx))
    #                 # test_reid() on self.model[idx] will only evaluate the 'mean_net'
    #                 # for testing 'net', use self.model[idx].module.net
    #                 cmc, mAP = test_reid(
    #                     cfg,
    #                     runner.model,
    #                     loader,
    #                     query,
    #                     gallery,
    #                     dataset_name=cfg.TEST.datasets[i],
    #                     visrankactiv= False,
    #                 )
    #                 print("map on "+l[i]+" domain : {}".format(mAP))
    #             if rank == 0:
    #                 # print("in rank 0 test")
    #                 writer.add_scalar("TEST_Map_Cam_"+str(r)+'_'+l[i]+"/"+str(idx),float(mAP),1)
    #aggregate
    models_0_net = [runners[i].model.module for i in range (len(runners))]
    if cfg.FED.fed_opt == "fedavg":
        weights_fed_avg = [i/sum(clients_sizes) for i in clients_sizes]
    elif cfg.FED.fed_opt == "fedfix":
        weights_fed_avg = None
    else:
        raise NotImplementedError("only support fedavg and fedfix")

    model_0_net_agg = aggregate_models(models_0_net, weights_fed_avg) #, cfg.FED.weights[models_0_net[0]]
    
    runner_agg = MMTRunner(
            cfg,
            [models_1[0],],
            optimizers[0],
            criterions_[0],
            train_loaders[0],
            writer,
            train_sets=train_setss[0],
            lr_scheduler=lr_schedulers[0],
            reset_optim=True,
        )
    # runner_agg.resume(cfg.work_dir / "model_best_agg.pth")
    runner_agg.model[0].module = model_0_net_agg
    runner_agg.model[0].module.net = model_0_net_agg.mean_net
    runner_agg._epoch = cfg.TRAIN.epochs - 1 
    torch.cuda.synchronize()
    if rank ==0:
        runner_agg.save(for_agg=True)
        # time.sleep(60)
    torch.cuda.synchronize()
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
            # for ru in range (len(runners)):
            # runner = runner_agg
            for idx in range(1):
                if cfg.TEST.datasets[i]=="msmt17" and i==0:
                    mAP = 0
                else:

                    print("==> Test on the no.{} model".format(idx))
                    # test_reid() on self.model[idx] will only evaluate the 'mean_net'
                    # for testing 'net', use self.model[idx].module.net
                    cmc, mAP = test_reid(
                        cfg,
                        runner_agg.model[idx],
                        loader,
                        query,
                        gallery,
                        dataset_name=cfg.TEST.datasets[i],
                        visrankactiv= False,
                    )
                    print("map on "+l[i]+" domain : {}".format(mAP))
                if rank == 0:
                    # print("in rank 0 test")
                    writer.add_scalar("TEST_Map_Server_"+l[i]+"/per_epoch",float(mAP),cfg.TRAIN.epochs)
                    writer.add_scalar("TEST_Map_Server_"+l[i]+"/per_round",float(mAP),1)

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))

if __name__ == "__main__":
    main()
