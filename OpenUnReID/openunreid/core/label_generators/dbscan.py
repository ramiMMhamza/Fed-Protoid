# Written by Yixiao Ge

import collections

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy import spatial
from ...utils.torch_utils import to_torch
from ..utils.compute_dist import build_dist

__all__ = ["label_generator_dbscan_single", "label_generator_dbscan"]


@torch.no_grad()
def label_generator_dbscan_single(cfg, features, dist_matrix, eps, **kwargs):
    assert isinstance(dist_matrix, np.ndarray)

    # clustering
    min_samples = cfg.TRAIN.PSEUDO_LABELS.min_samples
    use_outliers = cfg.TRAIN.PSEUDO_LABELS.use_outliers

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    # print(dist)
    labels = cluster.fit_predict(dist_matrix)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    centers = collections.defaultdict(list)
    outliers = 0
    count_outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                count_outliers += 1
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]
    centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters,count_outliers


@torch.no_grad()
def label_generator_dbscan(cfg, features,weighted_clustering=False, cuda=True, indep_thres=None, **kwargs):#features_source
    assert cfg.TRAIN.PSEUDO_LABELS.cluster == "dbscan"

    if not cuda:
        cfg.TRAIN.PSEUDO_LABELS.search_type = 3

    # compute distance matrix by features
    dist_matrix = build_dist(cfg.TRAIN.PSEUDO_LABELS, features, verbose=True)
    # if weighted_clustering : #and cfg.task_id>0:
    #     lambda_=cfg.TRAIN.PSEUDO_LABELS.lambda_
    #     tree = spatial.KDTree(features_source)
    #     Nold = tree.query(features)[0]
    #     # Nold = Nold.reshape(Nold.shape[0],1)
    #     # dist_weights = Nold.dot(Nold.T)
    #     Nold = Nold.reshape(1,Nold.shape[0])
    #     Nold = np.zeros((Nold.shape[1],1))+Nold
    #     Nold = np.exp(-Nold)
    #     dist_weights = Nold + Nold.T
    #     # dist_weights = 1/(1-np.exp(-lambda_*dist_weights))
    #     # dist = dist*dist_weights
    #     dist = (1-lambda_)*dist + lambda_*dist_weights
    #     print("new dist {}".format(dist))
    #     print("dist weights {}".format(dist_weights))
    # print("dist mean {}".format(dist.mean()))
    # print("dist var {}".format(dist.var()))
    # sys.exit()
    features = features.cpu()

    # clustering
    eps = cfg.TRAIN.PSEUDO_LABELS.eps

    if len(eps) == 1:
        # normal clustering
        labels, centers, num_classes, count_outliers = label_generator_dbscan_single(
            cfg, features, dist_matrix, eps[0]
        )
        return labels, centers, num_classes, indep_thres, 

    else:
        assert (
            len(eps) == 3
        ), "three eps values are required for the clustering reliability criterion"

        print("adopt the reliability criterion for filtering clusters")
        eps = sorted(eps)
        labels_tight, _, _ ,_= label_generator_dbscan_single(cfg, features, dist_matrix, eps[0])
        labels_normal, _, num_classes,_ = label_generator_dbscan_single(
            cfg, features, dist_matrix, eps[1]
        )
        labels_loose, _, _ ,_= label_generator_dbscan_single(cfg, features, dist_matrix, eps[2])

        # compute R_indep and R_comp
        N = labels_normal.size(0)
        label_sim = (
            labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float()
        )
        label_sim_tight = (
            labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float()
        )
        label_sim_loose = (
            labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float()
        )

        R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(
            label_sim, label_sim_tight
        ).sum(-1)
        R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(
            label_sim, label_sim_loose
        ).sum(-1)
        assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
        assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

        cluster_R_comp, cluster_R_indep = (
            collections.defaultdict(list),
            collections.defaultdict(list),
        )
        cluster_img_num = collections.defaultdict(int)
        for comp, indep, label in zip(R_comp, R_indep, labels_normal):
            cluster_R_comp[label.item()].append(comp.item())
            cluster_R_indep[label.item()].append(indep.item())
            cluster_img_num[label.item()] += 1

        cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
        cluster_R_indep = [
            min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
        ]
        cluster_R_indep_noins = [
            iou
            for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
            if cluster_img_num[num] > 1
        ]
        if indep_thres is None:
            indep_thres = np.sort(cluster_R_indep_noins)[
                min(
                    len(cluster_R_indep_noins) - 1,
                    np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
                )
            ]

        labels_num = collections.defaultdict(int)
        for label in labels_normal:
            labels_num[label.item()] += 1

        centers = collections.defaultdict(list)
        outliers = 0
        for i, label in enumerate(labels_normal):
            label = label.item()
            indep_score = cluster_R_indep[label]
            comp_score = R_comp[i]
            if label == -1:
                assert not cfg.TRAIN.PSEUDO_LABELS.use_outliers, "exists a bug"
                continue
            if (indep_score > indep_thres) or (
                comp_score.item() > cluster_R_comp[label]
            ):
                if labels_num[label] > 1:
                    labels_normal[i] = num_classes + outliers
                    outliers += 1
                    labels_num[label] -= 1
                    labels_num[labels_normal[i].item()] += 1

            centers[labels_normal[i].item()].append(features[i])

        num_classes += outliers
        assert len(centers.keys()) == num_classes

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)

        return labels_normal, centers, num_classes, indep_thres
