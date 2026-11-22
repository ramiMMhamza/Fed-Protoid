import contextlib
import io

import pytest
import torch
from torch.utils.data import DataLoader

from openunreid.models.backbones import build_bakcbone
from openunreid.models.builder import ReIDBaseModel, TeacherStudentNetwork
from openunreid.models.utils.extract import extract_features, extract_features_protos


@pytest.mark.parametrize("name", ["resnet18", "resnet34", "resnet50", "resnet101",
                                  "resnet152", "resnet50_ibn_a", "resnet101_ibn_a"])
def test_every_backbone_builds(name):
    assert isinstance(build_bakcbone(name, "unused.pth", pretrained=False), torch.nn.Module)


@pytest.fixture(scope="module")
def mean_teacher():
    torch.manual_seed(0)
    net = ReIDBaseModel("resnet18", "", 10, [], "gem", 0, 0.0, pretrained=False)
    return TeacherStudentNetwork(net, 0.999)


def test_training_forward_returns_student_and_teacher_outputs(mean_teacher):
    mean_teacher.train()
    student, teacher = mean_teacher(torch.randn(4, 3, 256, 128))
    for out in (student, teacher):
        assert out["prob"].shape == (4, 10)
        assert out["feat"].shape == out["pooling"].shape == (4, 512)


def test_eval_forward_is_the_teacher_and_ignores_protos_flag(mean_teacher):
    mean_teacher.eval()
    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        out = mean_teacher(x)
        assert torch.equal(out, mean_teacher.mean_net(x))
        assert torch.equal(out, mean_teacher(x, protos=True))


def test_prototype_extraction_matches_feature_extraction(mean_teacher):
    data = [{"img": torch.randn(3, 256, 128)} for _ in range(10)]
    loader = DataLoader(data, batch_size=4)
    dataset = [(f"img{i}", i // 2, 0) for i in range(10)]
    with contextlib.redirect_stdout(io.StringIO()):
        features, raw = extract_features_protos(mean_teacher, loader, dataset, cuda=False)
        reference = extract_features(mean_teacher, loader, dataset, cuda=False)
    assert torch.allclose(features, reference)
    assert torch.allclose(features, torch.nn.functional.normalize(raw, dim=1))
    assert not features.requires_grad
