"""Camera split of the target dataset."""
import contextlib
import io

import pytest
from easydict import EasyDict

from openunreid.data import build_train_dataloader
from openunreid.utils.config import cfg_from_yaml_file
from cpu_round import TOOLS


@pytest.fixture
def config(market_root):
    config = cfg_from_yaml_file(TOOLS / "FEDPROTO" / "config.yaml", EasyDict())
    config.DATA_ROOT = market_root
    config.total_gpus, config.n_tasks, config.task_id = 1, 6, -1
    config.TRAIN.iters, config.TRAIN.times_itc = 0, 0
    config.TRAIN.LOADER.samples_per_gpu = 8
    return config


def build(config, camera, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_train_dataloader(config, 6, camera, **kwargs)


@pytest.mark.parametrize("camera", range(6))
def test_a_client_only_sees_its_camera_without_labels(config, camera):
    _, sets = build(config, camera, filtre=[True, False], joint=False)
    data = sets[0].data
    assert len(data) == 12 * 6
    assert {cam for _, _, cam in data} == {camera}
    assert {pid for _, pid, _ in data} == {0}  # ground truth removed

