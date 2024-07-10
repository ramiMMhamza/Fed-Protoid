from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
import matplotlib.pyplot as plt
from sklearn import manifold
import io
import PIL.Image
from torchvision.transforms import ToTensor
import numpy as np
# writer = SummaryWriter('./out/logs_tb/',flush_secs=10)

n_tasks = 5
import argparse
import collections
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch, torchvision
from torch.nn.parallel import DataParallel, DistributedDataParallel

from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import (
    build_test_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from openunreid.core.metrics.accuracy import accuracy
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.models.utils.extract import extract_features, extract_features_for_similarities
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize, get_dist_info
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile



def import_pretrained_model(cfg, dist, init_):
    # build model
    model = build_model(
        cfg, 0, [], init=init_
    )  # use num_classes=0 since we do not need classifier for testing
    model.cuda()
    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=True,
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)
    for param in model.module.parameters():
        param.requires_grad = False
    model.train()

    return model
