import pytest
import torch
import torch.nn.functional as F

from openunreid.models.losses.classification import CrossEntropyLoss, MMD_loss
from openunreid.models.losses.triplet import SoftmaxTripletLoss, SoftSoftmaxTripletLoss


def test_mmd_is_zero_for_identical_samples_and_grows_with_the_shift():
    torch.manual_seed(0)
    x = F.normalize(torch.randn(16, 64), dim=1)
    mmd = MMD_loss()
    assert mmd(x, x).abs() < 1e-5
    near = mmd(x, F.normalize(x + 0.1 * torch.randn_like(x), dim=1))
    far = mmd(x, F.normalize(torch.randn(16, 64) + 3, dim=1))
    assert 0 < near < far


def test_mmd_needs_batches_of_the_same_size():
    # the four kernel blocks are summed element-wise; in training both batches
    # have `samples_per_gpu` elements
    mmd = MMD_loss()
    assert torch.isfinite(mmd(torch.randn(16, 32), torch.randn(16, 32)))
    with pytest.raises(RuntimeError):
        mmd(torch.randn(16, 32), torch.randn(8, 32))


def test_cross_entropy_with_label_smoothing():
    results = {"prob": torch.tensor([[10.0, 0.0, 0.0]])}
    loss, _ = CrossEntropyLoss(num_classes=3, epsilon=0.1)(results, torch.tensor([0]))
    assert 0 < loss < 1


def test_triplet_losses_run_and_are_finite():
    torch.manual_seed(0)
    targets = torch.arange(4).repeat_interleave(4)
    results = {"pooling": torch.randn(16, 32)}
    mean = {"pooling": torch.randn(16, 32)}
    assert torch.isfinite(SoftmaxTripletLoss(margin=0.0)(results, targets))
    assert torch.isfinite(SoftSoftmaxTripletLoss()(results, targets, mean))
