# Written by Yixiao Ge
from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import time
import warnings
from datetime import timedelta

import cv2
import numpy as np
import torch
import torchvision

from ..core.metrics.rank import evaluate_rank
from ..core.utils.compute_dist import build_dist
from ..models.utils.dsbn_utils import switch_target_bn
from ..models.utils.extract import extract_features
from ..data.utils.data_utils import save_image
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.file_utils import mkdir_if_missing
from ..utils.torch_utils import tensor2im
from ..utils.meters import Meters

# # Deprecated
# from ..core.utils.rerank import re_ranking_cpu
GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


@torch.no_grad()
def test_reid(
    cfg, model, data_loader, query, gallery, dataset_name=None, rank=None, visrankactiv=False, **kwargs
):

    start_time = time.monotonic()

    if cfg.MODEL.dsbn:
        assert (
            dataset_name is not None
        ), "the dataset_name for testing is required for DSBN."
        # switch bn for dsbn_based model
        if dataset_name in list(cfg.TRAIN.datasets.keys()):
            bn_idx = list(cfg.TRAIN.datasets.keys()).index(dataset_name)
            switch_target_bn(model, bn_idx)
        else:
            warnings.warn(
                f"the domain of {dataset_name} does not exist before, "
                f"the performance may be bad."
            )

    sep = "*******************************"
    if dataset_name is not None:
        print(f"\n{sep} Start testing {dataset_name} {sep}\n")

    if rank is None:
        rank, _, _ = get_dist_info()

    # parse ground-truth IDs and camera IDs
    q_pids = np.array([pid for _, pid, _ in query])
    g_pids = np.array([pid for _, pid, _ in gallery])
    q_cids = np.array([cid for _, _, cid in query])
    g_cids = np.array([cid for _, _, cid in gallery])

    # extract features with the given model
    if visrankactiv:
        features, activations = extract_features(
            model,
            data_loader,
            query + gallery,
            normalize=cfg.TEST.norm_feat,
            with_path=False,
            # one_gpu = one_gpu,
            prefix="Test: ",
            visrankactiv=visrankactiv,
            **kwargs,
        )
    else:
        features = extract_features(
            model,
            data_loader,
            query + gallery,
            normalize=cfg.TEST.norm_feat,
            with_path=False,
            # one_gpu = one_gpu,
            prefix="Test: ",
            **kwargs,
        )
    if rank == 0:
        # split query and gallery features
        assert features.size(0) == len(query) + len(gallery)
        query_features = features[: len(query)]
        gallery_features = features[len(query) :]
        if visrankactiv:
            # Split query and gallery activations
            qa = activations[: len(query)]
            ga = activations[len(query) :]
            # qa = torch.cat((qa, 0))
            # ga = torch.cat((ga, 0))

        # evaluate with original distance
        dist = build_dist(cfg.TEST, query_features, gallery_features)
        cmc, map = evaluate_rank(dist, q_pids, g_pids, q_cids, g_cids)
    else:
        cmc, map = np.empty(50), 0.0

    if cfg.TEST.rerank:

        # rerank with k-reciprocal jaccard distance
        print("\n==> Perform re-ranking")
        if rank == 0:
            rerank_dist = build_dist(
                cfg.TEST, query_features, gallery_features, dist_m="jaccard"
            )
            final_dist = (
                rerank_dist * (1 - cfg.TEST.lambda_value) + dist * cfg.TEST.lambda_value
            )
            # # Deprecated due to the slower speed
            # dist_qq = build_dist(cfg, query_features, query_features)
            # dist_gg = build_dist(cfg, gallery_features, gallery_features)
            # final_dist = re_ranking_cpu(dist, dist_qq, dist_gg)

            cmc, map = evaluate_rank(final_dist, q_pids, g_pids, q_cids, g_cids)
        else:
            cmc, map = np.empty(50), 0.0

    end_time = time.monotonic()
    print("Testing time: ", timedelta(seconds=end_time - start_time))
    print(f"\n{sep} Finished testing {sep}\n")
    if cfg.exec==0:
        save_dir = osp.join(cfg.work_dir, 'visrankactiv_'+str(cfg.task_id)+'_'+dataset_name)
    else:
        save_dir = '/out/visrankactiv_'+str(cfg.task_id)+'_'+dataset_name
    if visrankactiv and rank==0:
            visualize_ranked_activation_results(
                dist,
                qa,
                ga,
                (query,gallery),
                "image",
                width=128,
                height=256,
                save_dir= save_dir,
                topk=10
            )
    return cmc, map


@torch.no_grad()
def val_reid(
    cfg, model, data_loader, val, epoch=0, dataset_name=None, rank=None, **kwargs
):

    start_time = time.monotonic()

    sep = "*************************"
    if dataset_name is not None:
        print(f"\n{sep} Start validating {dataset_name} on epoch {epoch} {sep}n")

    if rank is None:
        rank, _, _ = get_dist_info()

    # parse ground-truth IDs and camera IDs
    pids = np.array([pid for _, pid, _ in val])
    cids = np.array([cid for _, _, cid in val])

    # extract features with the given model
    features = extract_features(
        model,
        data_loader,
        val,
        normalize=cfg.TEST.norm_feat,
        with_path=False,
        # one_gpu = one_gpu,
        prefix="Val: ",
        **kwargs,
    )

    # evaluate with original distance
    if rank == 0:
        dist = build_dist(cfg.TEST, features)
        cmc, map = evaluate_rank(dist, pids, pids, cids, cids)
    else:
        cmc, map = np.empty(50), 0.0
    end_time = time.monotonic()
    print("Validating time: ", timedelta(seconds=end_time - start_time))
    print(f"\n{sep} Finished validating {sep}\n")

    return cmc, map, features, pids


@torch.no_grad()
def infer_gan(
        cfg, model, data_loader, dataset_name=None, rank=None,
        cuda=True, print_freq=10, prefix="Translate: ", **kwargs
    ):

    start_time = time.monotonic()

    sep = "*******************************"
    if dataset_name is not None:
        print(f"\n{sep} Start translating {dataset_name} {sep}\n")

    if rank is None:
        rank, _, _ = get_dist_info()

    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)
    if rank == 0:
        mkdir_if_missing(osp.join(cfg.work_dir, dataset_name+'_translated'))

    model.eval()
    data_iter = iter(data_loader)

    end = time.time()
    for i in range(len(data_loader)):
        data = next(data_iter)
        progress.update({"Data": time.time() - end})

        images = data['img']
        if cuda:
            images = images.cuda()

        outputs = model(images)

        for idx in range(outputs.size(0)):
            save_path = os.path.join(cfg.work_dir, dataset_name+'_translated', osp.basename(data['path'][idx]))
            if (osp.isfile(save_path)):
                continue
            img_np = tensor2im(outputs[idx], mean=cfg.DATA.norm_mean, std=cfg.DATA.norm_std)
            save_image(img_np, save_path)

        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    synchronize()

    end_time = time.monotonic()
    print('Translating time: ', timedelta(seconds=end_time - start_time))
    print(f"\n{sep} Finished translating {sep}\n")

    return


def visualize_ranked_activation_results(distmat, query_act, gallery_act, dataset, data_type, width=128, height=256, save_dir='', topk=10):
    """Visualizes ranked results with activation maps.

    Supports only image-reid.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        query_act (torch tensor): activations for query (num_query)
        gallery_act (torch tensor): activations for gallery (num_gallery)
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    if data_type != 'image':
        raise KeyError("Unsupported data type: {}".format(data_type))
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))
    
    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)
    
    indices = np.argsort(distmat, axis=1)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones((2*height+10, num_cols*width+topk*GRID_SPACING+QUERY_EXTRA_SPACING, 3), dtype=np.uint8)

            qact = query_act[q_idx].numpy()
            qact = np.uint8(np.floor(qact))
            qact = cv2.applyColorMap(qact, cv2.COLORMAP_JET)
            overlapped = qimg * 0.5 + qact * 0.5
            overlapped[overlapped>255] = 255
            overlapped = overlapped.astype(np.uint8)
            grid_img[:height, :width, :] = qimg
            grid_img[height+10:, :width, :] = overlapped
        else:
            pass

        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            gimg_path, gpid, gcamid = gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid==qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx+1)*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING

                    gact = gallery_act[g_idx].numpy()
                    gact = np.uint8(np.floor(gact))
                    gact = cv2.applyColorMap(gact, cv2.COLORMAP_JET)
                    overlapped = gimg * 0.5 + gact * 0.5
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)
                    grid_img[:height, start: end, :] = gimg
                    grid_img[height+10:, start: end, :] = overlapped
                else:
                    pass

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname+'.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx+1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))