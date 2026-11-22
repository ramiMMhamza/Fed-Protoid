import math

import pytest

from cpu_round import run_client_round


@pytest.mark.slow
def test_one_client_round_runs_on_cpu(market_root, tmp_path):
    losses, _ = run_client_round(market_root, tmp_path)
    assert len(losses) == 4
    expected = {"cross_entropy", "soft_entropy", "softmax_triplet",
                "soft_softmax_triplet", "MMD_loss", "Acc@1"}
    for step in losses:
        assert set(step) == expected
        assert all(math.isfinite(v) for v in step.values())
    assert losses[0]["MMD_loss"] > 0
