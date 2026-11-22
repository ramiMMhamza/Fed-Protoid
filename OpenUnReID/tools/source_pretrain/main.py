"""Supervised pre-training on the source domain (the paper's "warm-up").

Produces the checkpoint used as MODEL.source_pretrained by the federated
training (Algorithm 1: "source pre-trained weights theta_s"). Table 2 of the
paper: the warm-up adds 9.5 mAP on MS->M for Fed-Protoid.

Based on the source pre-training of OpenUnReID (tools/source_pretrain); also
trains the ViT of Fed-Protoid++.

    python -m torch.distributed.run --nproc_per_node=2 \\
        OpenUnReID/tools/source_pretrain/main.py \\
        OpenUnReID/tools/source_pretrain/config.yaml \\
        --work-dir source_pretrain_msmt17 --launcher pytorch

The best model (validation mAP) is written to
LOGS_ROOT/<work-dir>/model_best.pth.
"""
import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from openunreid.apis import BaseRunner, set_random_seed, test_reid
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader
from openunreid.models import build_model, build_transformer
from openunreid.models.losses import build_loss
from openunreid.utils.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from openunreid.utils.dist_utils import get_dist_info, init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
from openunreid.utils.torch_utils import copy_state_dict

VIT_ARCHS = ("vit_small", "vit_base")


def parse_config():
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    parser.add_argument("config", help="config file")
    parser.add_argument("--work-dir", default="", help="sub-directory of LOGS_ROOT")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument("--data-root", help="override DATA_ROOT")
    parser.add_argument("--logs-root", help="override LOGS_ROOT")
    parser.add_argument("--backbone-path", help="override MODEL.backbone_path")
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER,
        help="override config keys, e.g. --set TRAIN.epochs 60",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    if args.data_root:
        cfg.DATA_ROOT = Path(args.data_root)
    if args.logs_root:
        cfg.LOGS_ROOT = Path(args.logs_root)
    if args.backbone_path:
        cfg.MODEL.backbone_path = args.backbone_path
    cfg.launcher, cfg.tcp_port = args.launcher, args.tcp_port
    # the data builders of this code base read the camera split from these
    cfg.n_tasks, cfg.task_id = None, -1
    cfg.work_dir = cfg.LOGS_ROOT / (args.work_dir or Path(args.config).stem)
    mkdir_if_missing(cfg.work_dir)
    shutil.copy(args.config, cfg.work_dir / "config.yaml")
    return args


def make_model(num_classes):
    """ResNet (Fed-Protoid) or ViT (Fed-Protoid++), without mean teacher."""
    if cfg.MODEL.backbone in VIT_ARCHS:
        # same construction as the federated Fed-Protoid++ scripts;
        # backbone_path holds the self-supervised LUPerson weights
        return build_transformer(
            cfg, cfg.MODEL.backbone, img_size=(cfg.DATA.height, cfg.DATA.width),
            drop_path_rate=0.3, pretrained_path=cfg.MODEL.backbone_path,
            hw_ratio=2, conv_stem=True, num_classes=num_classes,
        )
    return build_model(cfg, num_classes, [])


def unwrap(model):
    return model.module if hasattr(model, "module") else model


class SourceRunner(BaseRunner):
    """Keeps the checkpoint with the best validation mAP."""

    def val(self):
        mAP = super().val()
        if self._rank == 0:
            state = {"state_dict": unwrap(self.model).state_dict(),
                     "epoch": self._epoch + 1, "mAP": mAP}
            torch.save(state, self.cfg.work_dir / "checkpoint.pth")
            if mAP >= self._best_mAP:
                self._best_mAP = mAP
                torch.save(state, self.cfg.work_dir / "model_best.pth")
                print(f"=> new best model, mAP {mAP:.1%}, epoch {self._epoch + 1}")
        synchronize()
        return mAP


def main():
    start_time = time.monotonic()
    args = parse_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    rank, _, _ = get_dist_info()
    synchronize()

    sys.stdout = Logger(cfg.work_dir / "log.txt")
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)
    writer = SummaryWriter(cfg.work_dir / "logs_tb") if rank == 0 else None

    # labelled source data, no camera split
    train_loader, train_sets = build_train_dataloader(
        cfg, None, None, [False] * len(cfg.TRAIN.datasets)
    )
    num_classes = train_loader.loader.dataset.num_pids

    model = make_model(num_classes)
    model.cuda()
    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.gpu], output_device=cfg.gpu, find_unused_parameters=True
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    optimizer = build_optimizer([model], **cfg.TRAIN.OPTIM)
    lr_scheduler = (
        build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
        if cfg.TRAIN.SCHEDULER.lr_scheduler is not None else None
    )
    criterions = build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True)

    runner = SourceRunner(
        cfg, model, optimizer, criterions, train_loader, writer,
        train_sets=train_sets, lr_scheduler=lr_scheduler,
    )
    runner.run()

    # test the best model
    best = torch.load(cfg.work_dir / "model_best.pth", map_location="cpu")
    copy_state_dict(best["state_dict"], unwrap(model))
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        _, mAP = test_reid(cfg, model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i])
        print(f"test mAP on {cfg.TEST.datasets[i]}: {mAP:.1%}")

    print("Total running time: ", timedelta(seconds=time.monotonic() - start_time))


if __name__ == "__main__":
    main()
