from pathlib import Path

import pytest
from easydict import EasyDict

from openunreid.utils.config import cfg_from_yaml_file
from cpu_round import TOOLS

CONFIGS = [TOOLS / m / n for m in ("FEDPROTO", "FEDPROTO++") for n in ("config.yaml", "config_s.yaml")]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_config_paths_are_relative_and_complete(path):
    config = cfg_from_yaml_file(path, EasyDict())
    for value in (config.DATA_ROOT, config.LOGS_ROOT,
                  config.MODEL.backbone_path, config.MODEL.source_pretrained):
        assert not Path(value).is_absolute(), value
    for key in ("TRAIN", "TEST", "DATA", "MODEL"):
        assert key in config
    if path.name == "config.yaml":
        target = list(config.TRAIN.datasets)[0]
        assert len(config.FED.client_sizes[target]) == config.FED.num_cameras[target]


def test_loader_converts_the_paths_of_the_config_it_loads():
    config = cfg_from_yaml_file(CONFIGS[0], EasyDict())
    assert isinstance(config.DATA_ROOT, Path) and isinstance(config.LOGS_ROOT, Path)


def test_market_client_sizes_cover_the_training_set():
    config = cfg_from_yaml_file(CONFIGS[0], EasyDict())
    assert sum(config.FED.client_sizes["market1501"]) == 12936  # Market-1501 train images
