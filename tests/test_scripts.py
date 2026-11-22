"""The four training scripts."""
import pytest

import openunreid.fed
from cpu_round import TOOLS, load_tool

SCRIPTS = [(m, s) for m in ("FEDPROTO", "FEDPROTO++") for s in ("main_fedl_init.py", "main_fedl.py")]


@pytest.mark.parametrize("method, script", SCRIPTS)
def test_scripts_use_the_shared_aggregation(method, script):
    assert load_tool(method, script).aggregate_models is openunreid.fed.aggregate_models


@pytest.mark.parametrize("method, script", SCRIPTS)
def test_clients_only_use_their_own_images(method, script):
    # Eq. (6): the MMD loss uses the client's own mini-batch, so no loader
    # over the other cameras is built
    source = (TOOLS / method / script).read_text()
    assert "for_kd=True" not in source and "train_loader_target=" not in source


@pytest.mark.parametrize("method, script", SCRIPTS)
def test_prototypes_are_means_of_unnormalised_features(method, script):
    # Eq. (4): same features F(x) as on the target side of the MMD loss
    assert "normalize=False, with_path=False" in (TOOLS / method / script).read_text()
