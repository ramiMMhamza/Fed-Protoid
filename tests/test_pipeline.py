"""The whole pipeline on CPU: source pre-training, initial round, one round.

Deselected by default (about 4 minutes). Run with:  python -m pytest -m e2e
Small randomly initialised models and synthetic data: this checks that the
scripts run and hand over their checkpoints, not the accuracy.
"""
import re

import pytest
import torch
from easydict import EasyDict

from cpu_round import (make_market, make_msmt, run_training_script, source_pretrain,
                       write_small_configs)
from openunreid.models import build_model
from openunreid.utils.config import cfg_from_yaml_file


def loaded_into_resnet(config_path, checkpoint):
    """Build the model as the federated scripts do and check the weights."""
    config = cfg_from_yaml_file(config_path, EasyDict())
    model = build_model(config, 10, [], init=str(checkpoint))
    expected = torch.load(checkpoint, map_location="cpu")["state_dict"]
    actual = model.net.state_dict()
    return all(torch.equal(actual[k], v) for k, v in expected.items() if "classifier" not in k)


@pytest.mark.e2e
@pytest.mark.parametrize("method", ["FEDPROTO", "FEDPROTO++"])
def test_pretrain_then_initial_round_then_one_round(method, tmp_path):
    data = make_msmt(make_market(tmp_path / "data"))
    torch.save({}, tmp_path / "imagenet_or_lup.pth")  # stands in for the public weights

    source = source_pretrain(method, tmp_path, data, tmp_path / "imagenet_or_lup.pth")
    assert source.exists()

    config = write_small_configs(method, tmp_path, source)
    logs = tmp_path / "logs"
    first = run_training_script(method, "main_fedl_init.py", config, data, logs, "run")
    assert (logs / "run" / "model_best_agg.pth").exists()
    assert "map on target domain" in first

    # the federated scripts start from the source checkpoint
    if method == "FEDPROTO":
        assert loaded_into_resnet(config, source)
    else:
        loads = [int(a) for a, b in re.findall(r"Load (\d+) / (\d+) layers", first)]
        assert loads and min(loads) > 150  # of 178 ViT-S tensors (heads and BN neck excluded)

    second = run_training_script(method, "main_fedl.py", config, data, logs, "run")
    assert "aggregating models after round 1" in second
    assert "map on target domain" in second
    round_log = max((logs / "run").glob("log_*.txt"))
    # every client computed the MMD loss
    assert len(re.findall(r"MMD_loss [0-9.]+", round_log.read_text())) == 6

    if method == "FEDPROTO++":
        # the round resumes the whole aggregated model except the two
        # classifier heads (student and teacher)
        resumed = [(int(a), int(b)) for a, b in re.findall(r"Load (\d+) / (\d+) layers", second)
                   if int(b) > 300]
        assert resumed and all(total - loaded == 2 for loaded, total in resumed)
