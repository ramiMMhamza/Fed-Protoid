# Written by Yixiao Ge

import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data import build_val_dataloader, build_val_dataloader_source
from ...models.utils.extract import extract_features
from ...utils.dist_utils import (
    broadcast_tensor,
    broadcast_value,
    get_dist_info,
    synchronize,
)
from .dbscan import label_generator_dbscan, label_generator_dbscan_single  # noqa
from .kmeans import label_generator_kmeans
# from OpenUnReID.openunreid import data


class LabelGenerator(object):
    """Pseudo Label Generator."""

    __factory = {
        "dbscan": label_generator_dbscan,
        "kmeans": label_generator_kmeans,
    }

    def __init__(
        self, cfg, models, verbose=True  # list of models, e.g. MMT has two models
    ):
        super(LabelGenerator, self).__init__()

        assert (
            "PSEUDO_LABELS" in cfg.TRAIN
        ), "cannot find settings in the config file for pseudo labels"

        self.cfg = cfg
        if isinstance(models, nn.Module):
            models = [models]
        self.models = models
        self.verbose = verbose

        self.data_loaders, self.datasets = build_val_dataloader(
            cfg, for_clustering=True
        )

        # self.data_loaders_source, self.datasets_source = build_val_dataloader_source(
        #     cfg,task_id=cfg.task_id-1,for_clustering=True
        # )
# ,task_id=cfg.task_id-1
        self.cluster_type = self.cfg.TRAIN.PSEUDO_LABELS.cluster

        self.num_classes = []
        self.indep_thres = []

        if self.cfg.TRAIN.PSEUDO_LABELS.cluster_num is not None:
            # for kmeans
            self.num_classes = [self.cfg.TRAIN.PSEUDO_LABELS.cluster_num]
        self.num_clusters = 0
        self.rank, self.world_size, _ = get_dist_info()
        
    @torch.no_grad()
    def __call__(self, epoch, cuda=True, memory_features=None, **kwargs):
        # print(memory_features)
        all_labels = []
        all_centers = []
        # all_features = []
        # for idx, (data_loader_s, dataset_s) in enumerate(
        #     zip(self.data_loaders_source, self.datasets_source)
        # ):
        #     all_features_source = []
        #     for model in self.models:
        #             features = extract_features(
        #                 model,
        #                 data_loader_s,
        #                 dataset_s,
        #                 cuda,
        #                 normalize=self.cfg.TRAIN.PSEUDO_LABELS.norm_feat,
        #                 with_path=False,
        #                 for_testing=False,
        #                 prefix="Cluster: ",
        #                 **kwargs,
        #             )
        #             all_features_source.append(features)
        # print(self.data_loaders)
        for idx, (data_loader, dataset) in enumerate(
            zip(self.data_loaders, self.datasets)
        ):

            # clustering
            try:
                indep_thres = self.indep_thres[idx]
            except Exception:
                indep_thres = None
            try:
                num_classes = self.num_classes[idx]
            except Exception:
                num_classes = None

            if memory_features is None:
                # extract features
                all_features = []
                
                for model in self.models:
                    features = extract_features(
                        model,
                        data_loader,
                        dataset,
                        cuda,
                        normalize=self.cfg.TRAIN.PSEUDO_LABELS.norm_feat,
                        with_path=False,
                        for_testing=False,
                        prefix="Cluster: ",
                        **kwargs,
                    )
                    all_features.append(features)
                
                # all_features = self.adapting_features(epoch,all_features,self.cfg.TRAIN.PSEUDO_LABELS.mode)
                # all_features_source = self.adapting_features(epoch,all_features_source,self.cfg.TRAIN.PSEUDO_LABELS.mode)
                # all_features[0] = all_features[0]*1
                # if epoch<20:
                #     # all_features[0] = all_features[0]*(-0.04*epoch+1)
                #     all_features[1] = all_features[1]*(0.04*epoch+0.24)
                # else:
                #     # all_features[0] = all_features[0]*(-0.04*(epoch%20)+1)
                #     all_features[1] = all_features[1]*(0.04*(epoch%20)+0.24)
                all_features = torch.stack(all_features, dim=0).mean(0)
                # all_features_source = torch.stack(all_features_source, dim=0).mean(0)

                if "num_parts" in self.cfg.MODEL:
                    if self.cfg.MODEL.num_parts > 1:
                        num_splits = self.cfg.MODEL.num_parts
                    if self.cfg.MODEL.include_global:
                        num_splits += 1
                    all_features = torch.split(
                        all_features, all_features.size(1) // num_splits, dim=1
                    )
                    # all_features_source = torch.split(
                    #     all_features_source, all_features_source.size(1) // num_splits, dim=1
                    # )

            else:
                assert isinstance(memory_features, list)
                all_features = memory_features[idx]

            if self.cfg.TRAIN.PSEUDO_LABELS.norm_feat:
                if isinstance(all_features, list):
                    all_features = [F.normalize(f, p=2, dim=1) for f in all_features]
                    # all_features_source = [F.normalize(f, p=2, dim=1) for f in all_features_source]
                else:
                    all_features = F.normalize(all_features, p=2, dim=1)
                    # all_features_source = F.normalize(all_features_source, p=2, dim=1)

            if self.rank == 0:
                # clustering only on GPU:0
                labels, centers, num_classes, indep_thres = self.__factory[
                    self.cluster_type
                ](
                    self.cfg,
                    all_features,
                    # all_features_source,
                    weighted_clustering=self.cfg.TRAIN.PSEUDO_LABELS.weighted_clustering,
                    num_classes=num_classes,
                    cuda=cuda,
                    indep_thres=indep_thres,
                )

                if self.cfg.TRAIN.PSEUDO_LABELS.norm_center:
                    centers = F.normalize(centers, p=2, dim=1)

            synchronize()

            # broadcast to other GPUs
            if self.world_size > 1:
                num_classes = int(broadcast_value(num_classes, 0))
                if (
                    self.cfg.TRAIN.PSEUDO_LABELS == "dbscan"
                    and len(self.cfg.TRAIN.PSEUDO_LABELS.eps) > 1
                ):
                    # use clustering reliability criterion
                    indep_thres = broadcast_value(indep_thres, 0)
                if self.rank > 0:
                    labels = torch.arange(len(dataset)).long()
                    centers = torch.zeros((num_classes, all_features.size(-1))).float()
                labels = broadcast_tensor(labels, 0)
                centers = broadcast_tensor(centers, 0)

            try:
                self.indep_thres[idx] = indep_thres
            except Exception:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except Exception:
                self.num_classes.append(num_classes)

            all_labels.append(labels.tolist())
            all_centers.append(centers)

        self.cfg.TRAIN.PSEUDO_LABELS.cluster_num = self.num_classes

        if self.verbose:
            dataset_names = [
                list(self.cfg.TRAIN.datasets.keys())[i]
                for i in self.cfg.TRAIN.unsup_dataset_indexes
            ]
            for label, dn in zip(all_labels, dataset_names):
                self.print_label_summary(epoch, label, dn)
        # print(all_labels)
        return all_labels, all_centers, self.num_clusters, all_features

    def print_label_summary(self, epoch, pseudo_labels, dataset_name):
        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label] += 1
        if -1 in index2label.keys():
            unused_ins_num = index2label.pop(-1)
        else:
            unused_ins_num = 0
        index2label = np.array(list(index2label.values()))
        clu_num = (index2label > 1).sum()
        unclu_ins_num = (index2label == 1).sum()
        print(
            f"\n==> Statistics for {dataset_name} on epoch {epoch}: "
            f"{clu_num} clusters, "
            f"{unclu_ins_num} un-clustered instances, "
            f"{unused_ins_num} unused instances\n"
        )
        self.num_clusters = clu_num
    
    def adapting_features(self,epoch,all_features,mode) : 
        
        if mode == 'only_source' :
            if epoch%self.cfg.TRAIN.epochs==0:
                all_features[0] = all_features[0]*1
                all_features[1] = all_features[1]*0
            else:
                all_features[0] = all_features[0]*0
                all_features[1] = all_features[1]*1
        elif mode == 'coupled_1st_epoch_only_source':
            if epoch%self.cfg.TRAIN.epochs==0:
                all_features[0] = all_features[0]*1
                all_features[1] = all_features[1]*0
        elif mode == 'coupled_linear_on_source' :
            taux = 1/(self.cfg.TRAIN.epochs-1)
            beta = 1
            all_features[0] = all_features[0]*(-1*taux*(epoch%self.cfg.TRAIN.epochs) + beta)
        elif mode == 'coupled_linear' : 
            taux = 1/(self.cfg.TRAIN.epochs-1)
            beta = 1
            all_features[0] = all_features[0]*(-1*taux*(epoch%self.cfg.TRAIN.epochs) + beta)
            all_features[1] = all_features[1]*(1*taux*(epoch%self.cfg.TRAIN.epochs))
        elif mode == 'coupled_sigmoid' :
            all_features[0] = all_features[0]*(1 - (1/(1+np.exp(-epoch%self.cfg.TRAIN.epochs))))
            all_features[1] = all_features[1]*(1/(1+np.exp(-epoch%self.cfg.TRAIN.epochs)))
        elif mode == 'coupled_sigmoid_mod' :
            all_features[0] = all_features[0]*(1.5 - (1/(1+np.exp(-epoch%self.cfg.TRAIN.epochs))))
            all_features[1] = all_features[1]*(1/(1+np.exp(-epoch%self.cfg.TRAIN.epochs))-0.5)
        elif mode == "normal":
            all_features=[all_features[1]]
        return all_features
