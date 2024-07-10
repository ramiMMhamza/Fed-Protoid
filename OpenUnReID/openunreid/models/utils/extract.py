# Written by Yixiao Ge

import time
from collections import OrderedDict
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...utils.dist_utils import all_gather_tensor, get_dist_info, synchronize
from ...utils.meters import Meters


@torch.no_grad()
def extract_features(
    model,  # model used for extracting
    data_loader,  # loading data
    dataset,  # dataset with file paths, etc
    cuda=True,  # extract on GPU
    normalize=True,  # normalize feature
    with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
    print_freq=10,  # log print frequence
    save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
    for_testing=True,
    prefix="Extract: ",
    visrankactiv=False,
):
 
 
    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)

    rank, world_size, is_dist = get_dist_info()
    features = []
    activations = []
    model.eval()
    data_iter = iter(data_loader)

    end = time.time()
    for i in range(len(data_loader)):
        data = next(data_iter)
        progress.update({"Data": time.time() - end})
        images = data["img"]
        if cuda:
            images = images.cuda()

        # compute output
        outputs = model(images)
        if visrankactiv:
            activation = extract_activations(model,images)
            activation =  torch.Tensor(activation)
            activations.append(activation)
        if isinstance(outputs, list) and for_testing:
            outputs = torch.cat(outputs, dim=1)

        if normalize:
            if isinstance(outputs, list):
                outputs = [F.normalize(out, p=2, dim=-1) for out in outputs]
            outputs = F.normalize(outputs, p=2, dim=-1)

        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=1).data.cpu()
        else:
            outputs = outputs.data.cpu()
        
        features.append(outputs)

        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    synchronize()
    if is_dist and cuda:
        # distributed: gather features from all GPUs
        features = torch.cat(features)
        all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
        all_features = all_features.cpu()[: len(dataset)]
        if visrankactiv:
            activations = torch.cat(activations)
            all_activations = all_gather_tensor(activations.cuda(), save_memory=save_memory)
            all_activations = all_activations.cpu()[: len(dataset)]

    else:
        # no distributed, no gather
        all_features = torch.cat(features, dim=0)[: len(dataset)]
        if visrankactiv:
            all_activations = torch.cat(activations, dim=0)[:len(dataset)]
            
    if not with_path:
        if visrankactiv:
            return all_features, all_activations
        else:
            return all_features

    features_dict = OrderedDict()
    for fname, feat in zip(dataset, all_features):
        features_dict[fname[0]] = feat
    if visrankactiv:
        activations_dict = OrderedDict()
        for fname, activ in zip(dataset, all_activations):
            activations_dict[fname[0]] = activ
        return features_dict, activations_dict
    else:
        return features_dict

def extract_features_protos(
    model,  # model used for extracting
    data_loader,  # loading data
    dataset,  # dataset with file paths, etc
    cuda=True,  # extract on GPU
    normalize=True,  # normalize feature
    with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
    print_freq=10,  # log print frequence
    save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
    for_testing=True,
    prefix="Extract: ",
    visrankactiv=False,
    protos=False
):
 
 
    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)

    rank, world_size, is_dist = get_dist_info()
    features = []
    activations = []
    protos_ = []
    model.eval()
    data_iter = iter(data_loader)

    end = time.time()
    for i in range(len(data_loader)):
        data = next(data_iter)
        progress.update({"Data": time.time() - end})
        images = data["img"]
        if cuda:
            images = images.cuda()

        # compute output
        outputs = model(images)
        protos = model(images, return_featuremaps=False, protos=True)

        if isinstance(outputs, list) and for_testing:
            outputs = torch.cat(outputs, dim=1)
            protos = torch.cat(protos, dim=1)

        if normalize:
            if isinstance(outputs, list):
                outputs = [F.normalize(out, p=2, dim=-1) for out in outputs]
                # protos = [F.normalize(out, p=2, dim=-1) for out in protos]
            outputs = F.normalize(outputs, p=2, dim=-1)
            # protos = F.normalize(protos, p=2, dim=-1)

        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=1).data.cpu()
            protos = torch.cat(outputs, dim=1).data.cpu()
        else:
            outputs = outputs.data.cpu()
            protos = protos.data.cpu()
        
        features.append(outputs)
        protos_.append(protos)

        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    synchronize()
    if is_dist and cuda:
        # distributed: gather features from all GPUs
        features = torch.cat(features)
        all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
        all_features = all_features.cpu()[: len(dataset)]

        protos_ = torch.cat(protos_)
        all_protos_ = all_gather_tensor(protos_.cuda(), save_memory=save_memory)
        all_protos_ = all_protos_.cpu()[: len(dataset)]

    else:
        # no distributed, no gather
        all_features = torch.cat(features, dim=0)[: len(dataset)]
        all_protos_ = torch.cat(protos_, dim=0)[: len(dataset)]
        if visrankactiv:
            all_activations = torch.cat(activations, dim=0)[:len(dataset)]
            
    if not with_path:
        if visrankactiv:
            return all_features, all_activations
        else:
            return all_features, all_protos_

    features_dict = OrderedDict()
    for fname, feat in zip(dataset, all_features):
        features_dict[fname[0]] = feat
    if visrankactiv:
        activations_dict = OrderedDict()
        for fname, activ in zip(dataset, all_activations):
            activations_dict[fname[0]] = activ
        return features_dict, activations_dict
    else:
        return features_dict

def extract_features_for_similarities(
    model,  # model used for extracting
    data_loader,  # loading data
    dataset,  # dataset with file paths, etc
    cuda=True,  # extract on GPU
    normalize=True,  # normalize feature
    with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
    print_freq=10,  # log print frequence
    save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
    for_testing=True,
    prefix="Extract: ",
    indices = [1,5,10,15,20,25,48,150]
):

    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)

    rank, world_size, is_dist = get_dist_info()
    features_target = []
    inds_target = []
    features_source = []
    inds_source = []
    features = {"target": {} , "source":{}}
    model.eval()
    data_loader.new_epoch(0)

    end = time.time()
    for i in range(len(data_loader)):
        data =  data_loader.next()
        # if cuda:
        #     data = data.cuda()
        progress.update({"Data": time.time() - end})
        images_target = data[0]["img"]
        ind_target = data[0]["ind"]
        images_source = data[1]["img"]
        ind_source = data[1]["ind"]
        if len(images_target)==2:
            images_target = images_target[0]
            images_source = images_source[0]

        if cuda:
            images_target = images_target.cuda()
            images_source = images_source.cuda()

        # compute output
        outputs_target = model(images_target)
        outputs_source = model(images_source)

        if isinstance(outputs_target, list) and for_testing:
            outputs_target = torch.cat(outputs_target, dim=1)
            outputs_source = torch.cat(outputs_source, dim=1)

        if normalize:
            if isinstance(outputs_target, list):
                outputs_target = [F.normalize(out, p=2, dim=-1) for out in outputs_target]
                outputs_source = [F.normalize(out, p=2, dim=-1) for out in outputs_source]
            outputs_target = F.normalize(outputs_target, p=2, dim=-1)
            outputs_source = F.normalize(outputs_source, p=2, dim=-1)

        if isinstance(outputs_target, list):
            outputs_target = torch.cat(outputs_target, dim=1).data.cpu()
            outputs_source = torch.cat(outputs_source, dim=1).data.cpu()
        else:
            outputs_target = outputs_target.data.cpu()
            outputs_source = outputs_source.data.cpu()

        inds_target.append([ts.item() for ts in ind_target])
        inds_source.append([ts.item() for ts in ind_source])
        features_target.append(outputs_target)
        features_source.append(outputs_source)
        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
    
    synchronize()
    for i in range(len(features_source)):
        batch_features_target = features_target[i]
        batch_features_source = features_source[i]
        batch_inds_target = inds_target[i]
        batch_inds_source = inds_source[i]
        for j in range(len(batch_features_target)):
            features["target"][batch_inds_target[j]] = batch_features_target[j]
            features["source"][batch_inds_source[j]] = batch_features_source[j]
    # if is_dist and cuda:
    #     # distributed: gather features from all GPUs
    #     features = torch.cat(features)
    #     all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
    #     all_features = all_features.cpu()[: len(dataset)]

    # else:
    #     # no distributed, no gather
    #     all_features = torch.cat(features, dim=0)[: len(dataset)]

    # if not with_path:
    return features

    # features_dict = OrderedDict()
    # for fname, feat in zip(dataset, all_features):
    #     features_dict[fname[0]] = feat

    # return features_dict

@torch.no_grad()
def extract_activations(model, input):
    model.eval()
    outputs = model(input, return_featuremaps=True)
    print(outputs.size())
    outputs = (outputs**2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h*w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)
    activations = []
    for j in range(outputs.size(0)):
        # activation map
        am = outputs[j, ...].cpu().numpy()
        am = cv2.resize(am, (128, 256))
        am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
        activations.append(am)
    return np.array(activations)

@torch.no_grad()
def extract_poolings(model, input):
    model.eval()
    outputs = model(input, protos=True)
    return outputs