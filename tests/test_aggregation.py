"""Server aggregation, Eq. (2) of the paper (openunreid/fed/aggregation.py)."""
import copy

import pytest
import torch

from openunreid.fed import aggregate_models, is_classifier_head
from openunreid.models.builder import ReIDBaseModel, TeacherStudentNetwork


def make(num_classes, shift):
    torch.manual_seed(num_classes)
    model = TeacherStudentNetwork(
        ReIDBaseModel("resnet18", "", num_classes, [], "gem", 0, 0.0, pretrained=False)
    )
    for name, buf in model.named_buffers():
        if "running_mean" in name:
            buf.fill_(shift)
    for p in model.parameters():
        p.data.add_(0.01 * shift)
    return model


@pytest.fixture
def models():
    return [make(10, 1.0), make(7, 2.0), make(5, 3.0)]  # source, client 1, client 2


def tensor(model, name="net.backbone.conv1.weight"):
    return model.state_dict()[name]


def test_fedavg_weights(models):
    ref = copy.deepcopy(models)
    agg = aggregate_models(models, [0.25, 0.75], alpha=0.2)
    expected = 0.2 * tensor(ref[0]) + 0.8 * (0.25 * tensor(ref[1]) + 0.75 * tensor(ref[2]))
    assert torch.allclose(tensor(agg), expected, atol=1e-6)


def test_fedfix_weights(models):
    ref = copy.deepcopy(models)
    agg = aggregate_models(models, None, alpha=0.2)
    expected = 0.2 * tensor(ref[0]) + 0.4 * tensor(ref[1]) + 0.4 * tensor(ref[2])
    assert torch.allclose(tensor(agg), expected, atol=1e-6)


def test_bn_statistics_are_averaged_like_the_weights(models):
    agg = aggregate_models(models, [0.5, 0.5], alpha=0.2)
    stats = tensor(agg, "net.backbone.bn1.running_mean")
    assert torch.allclose(stats, torch.full_like(stats, 0.2 * 1 + 0.8 * (0.5 * 2 + 0.5 * 3)))


def test_original_behaviour_for_bn_statistics_is_available(models):
    agg = aggregate_models(models, [0.5, 0.5], alpha=0.2, average_buffers=False)
    assert torch.all(tensor(agg, "net.backbone.bn1.running_mean") == 3.0)  # last client


def test_classifier_heads_stay_personal(models):
    ref = copy.deepcopy(models)
    agg = aggregate_models(models, [0.5, 0.5])
    assert agg is models[-1]
    assert torch.equal(agg.net.classifier.weight, ref[2].net.classifier.weight)


def test_only_the_returned_model_is_modified(models):
    ref = copy.deepcopy(models)
    aggregate_models(models, [0.5, 0.5])
    for model, before in zip(models[:-1], ref[:-1]):
        state = model.state_dict()
        assert all(torch.equal(state[k], v) for k, v in before.state_dict().items())


@pytest.mark.parametrize("name, head", [
    ("net.classifier.weight", True),         # ResNet classifier
    ("mean_net.fc.weight", True),            # ViT classifier
    ("net.blocks.3.mlp.fc1.weight", False),  # ViT MLP, not a head
    ("net.blocks.3.mlp.fc2.bias", False),
    ("net.bottleneck.weight", False),
    ("net.head.feat_bn.running_mean", False),
])
def test_classifier_head_detection(name, head):
    assert is_classifier_head(name) is head


def test_weights_are_checked(models):
    with pytest.raises(ValueError):
        aggregate_models(models, [0.5, 0.6])
    with pytest.raises(ValueError):
        aggregate_models(models, [1.0])


def test_server_sends_the_aggregated_teacher_as_student_and_teacher(models):
    # Sec. 3.2 and Table 3: teachers are transmitted, and both models of the
    # next round start from the aggregated one (main_fedl*.py: net = mean_net)
    agg = aggregate_models(models, [0.5, 0.5])
    agg.net = agg.mean_net
    state = agg.state_dict()
    for key in (k for k in state if k.startswith("net.")):
        assert torch.equal(state[key], state[key.replace("net.", "mean_net.", 1)])
