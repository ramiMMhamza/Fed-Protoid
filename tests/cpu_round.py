"""Run the Fed-Protoid client training on CPU, with a synthetic dataset.

The training code calls ``.cuda()`` everywhere. ``on_cpu()`` turns those calls
into no-ops so the real code path (clustering, sampler, losses, runner,
validation) can run without a GPU. Used by ``tests/test_round.py`` and as a
regression check for refactoring:

    python tests/cpu_round.py save before.json     # on the reference commit
    python tests/cpu_round.py compare before.json  # after the change

``compare`` exits with status 1 if any loss value or any weight differs.
The numbers depend on the PyTorch version and the CPU, so compare two commits
in the same environment.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import random
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "OpenUnReID" / "tools"


@contextlib.contextmanager
def on_cpu():
    """Make ``.cuda()`` a no-op for tensors and modules."""
    saved = torch.Tensor.cuda, torch.nn.Module.cuda, torch.cuda.synchronize
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.cuda.synchronize = lambda *args, **kwargs: None
    try:
        yield
    finally:
        torch.Tensor.cuda, torch.nn.Module.cuda, torch.cuda.synchronize = saved


def make_market(root, n_pid=12, n_cam=6, n_img=6, seed=0):
    """Write a Market-1501-like tree under ``root``.

    One flat colour per identity, plus noise per image, so that even a randomly
    initialised network separates the identities and DBSCAN finds clusters.
    """
    rng = np.random.default_rng(seed)
    base_dir = Path(root) / "market1501" / "Market-1501-v15.09.15"
    for split in ("bounding_box_train", "bounding_box_test", "query"):
        (base_dir / split).mkdir(parents=True, exist_ok=True)
    for pid in range(1, n_pid + 1):
        colour = np.ones((32, 16, 3)) * rng.integers(0, 256, 3)
        for cam in range(1, n_cam + 1):
            for k in range(n_img):
                img = np.clip(colour + rng.normal(0, 8, colour.shape), 0, 255)
                Image.fromarray(img.astype(np.uint8)).resize((64, 128)).save(
                    base_dir / "bounding_box_train" / f"{pid:04d}_c{cam}s1_{k:06d}_00.jpg"
                )
    # test identities: one query per identity (camera 1), gallery on cameras 2-6
    for pid in range(n_pid + 1, n_pid + 7):
        colour = np.ones((32, 16, 3)) * rng.integers(0, 256, 3)
        for cam in range(1, n_cam + 1):
            img = np.clip(colour + rng.normal(0, 8, colour.shape), 0, 255)
            split = "query" if cam == 1 else "bounding_box_test"
            Image.fromarray(img.astype(np.uint8)).resize((64, 128)).save(
                base_dir / split / f"{pid:04d}_c{cam}s1_000000_00.jpg"
            )
    return Path(root)


def make_msmt(root, n_pid=40, n_cam=3, n_img=2, n_test=6, seed=1):
    """Write an MSMT17-like tree (the source domain) under ``root``."""
    rng = np.random.default_rng(seed)
    base_dir = Path(root) / "msmt17" / "MSMT17_V1"
    lists = {"train": [], "val": [], "query": [], "gallery": []}
    for pid in range(n_pid + n_test):
        colour = np.ones((32, 16, 3)) * rng.integers(0, 256, 3)
        split = "train" if pid < n_pid else "test"
        for cam in range(1, n_cam + 1):
            for k in range(n_img):
                name = f"{pid:04d}/{pid:04d}_{k:03d}_{cam:02d}_0303morning_{k:04d}_0.jpg"
                path = base_dir / split / name
                path.parent.mkdir(parents=True, exist_ok=True)
                img = np.clip(colour + rng.normal(0, 8, colour.shape), 0, 255)
                Image.fromarray(img.astype(np.uint8)).resize((64, 128)).save(path)
                if split == "train":
                    lists["val" if pid % 5 == 0 else "train"].append(f"{name} {pid}")
                else:
                    lists["query" if cam == 1 else "gallery"].append(f"{name} {pid}")
    for name, lines in lists.items():
        (base_dir / f"list_{name}.txt").write_text("\n".join(lines) + "\n")
    return Path(root)


def load_tool(method="FEDPROTO", script="main_fedl_init.py"):
    """Import one of the training scripts as a module."""
    spec = importlib.util.spec_from_file_location(
        f"{method}_{script[:-3]}".replace("+", "p"), TOOLS / method / script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    from openunreid.data import builder

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    builder.g.manual_seed(0)  # module-level generator of the data loaders


def run_client_round(data_root, work_dir, camera=0, seed=0):
    """Train the client of ``camera`` for one round, as main_fedl_init.py does.

    Differences with a real run, to keep it small: ResNet-18 without ImageNet
    weights instead of ResNet-50 initialised from the source checkpoint, a
    batch of 8, 32 random prototypes, CPU clustering (``search_type: 3``).

    Returns ``(losses, model)``: the meters of every training step and the
    trained model.
    """
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    from openunreid.core.solvers import build_optimizer
    from openunreid.data import build_train_dataloader
    from openunreid.models import build_model
    from openunreid.models.losses import build_loss
    from openunreid.utils.config import cfg, cfg_from_yaml_file

    fed = load_tool()
    cfg.clear()
    cfg_from_yaml_file(TOOLS / "FEDPROTO" / "config.yaml", cfg)
    cfg.DATA_ROOT, cfg.work_dir = Path(data_root), Path(work_dir)
    cfg.MODEL.backbone, cfg.MODEL.imagenet_pretrained = "resnet18", False
    cfg.TRAIN.PSEUDO_LABELS.search_type = 3
    cfg.TRAIN.LOADER.samples_per_gpu = 8
    cfg.TRAIN.iters, cfg.TRAIN.times_itc, cfg.TRAIN.epochs = 0, 4, 1
    cfg.total_gpus, cfg.n_tasks, cfg.task_id = 1, 6, camera
    seed_everything(seed)

    losses = []
    with on_cpu(), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaders, sets = build_train_dataloader(cfg, 6, camera, [True, False], joint=False)
        num_classes = len(sets[0])
        model = torch.nn.DataParallel(build_model(cfg, num_classes, []))
        prototypes = torch.nn.functional.normalize(torch.randn(32, 512), dim=1)
        runner = fed.MMTRunner(
            cfg,
            model,
            build_optimizer([model], **cfg.TRAIN.OPTIM),
            build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True),
            [loaders[0]],
            SummaryWriter(Path(work_dir) / "tb"),
            train_sets=[sets[0]],
            reset_optim=True,
            protos_dataloader=DataLoader(
                TensorDataset(prototypes), batch_size=8, drop_last=True
            ),
        )
        update = runner.train_progress.update

        def record(meters):
            if "Acc@1" in meters:  # one call per training step
                losses.append({k: float(v) for k, v in meters.items()})
            update(meters)

        runner.train_progress.update = record
        runner.run(cam_id=camera)
    return losses, model


def write_small_configs(method, out_dir, source_ckpt):
    """Copies of the method's configs, shrunk to run in a few minutes on CPU."""
    import yaml

    paths = {}
    for name in ("config.yaml", "config_s.yaml"):
        config = yaml.safe_load((TOOLS / method / name).read_text())
        config["MODEL"].update(backbone="resnet18", imagenet_pretrained=False,
                               source_pretrained=str(source_ckpt))
        config["TRAIN"]["LOADER"].update(samples_per_gpu=4, workers_per_gpu=0)  # x2 "GPUs"
        config["TEST"]["LOADER"]["workers_per_gpu"] = 0
        if name == "config.yaml":  # synthetic Market-1501 as target, for both methods
            config["TRAIN"].update(datasets={"market1501": "trainval"}, val_dataset="market1501")
            config["TEST"]["datasets"] = ["msmt17", "market1501"]
        if "PSEUDO_LABELS" in config["TRAIN"]:
            config["TRAIN"]["PSEUDO_LABELS"]["search_type"] = 3  # CPU faiss
        paths[name] = Path(out_dir) / name
        paths[name].write_text(yaml.safe_dump(config))
    return paths["config.yaml"]


def run_script(script, *args):
    """Run a script of tools/ in a subprocess, on CPU. Returns its output."""
    import os
    import subprocess

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1", PYTHONWARNINGS="ignore")
    cmd = [sys.executable, __file__, "run-script", str(script), *map(str, args)]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    return result.stdout


def run_training_script(method, script, config, data_root, logs_root, run_name, *extra):
    """Run tools/<method>/<script> for one short round. Returns its output."""
    return run_script(
        TOOLS / method / script, config, f"--work-dir={run_name}", f"--data-root={data_root}",
        f"--logs-root={logs_root}", "--epochs_s", "1", "--epochs_t", "1", "--times_itc", "1",
        *extra,
    )


def source_pretrain(method, out_dir, data_root, backbone_ckpt):
    """Run tools/source_pretrain for a few iterations. Returns the checkpoint path."""
    import yaml

    name = "config_vit.yaml" if method == "FEDPROTO++" else "config.yaml"
    config = yaml.safe_load((TOOLS / "source_pretrain" / name).read_text())
    if method == "FEDPROTO":
        config["MODEL"].update(backbone="resnet18", imagenet_pretrained=False)
    config["TRAIN"].update(epochs=2, iters=4, val_freq=1)
    config["TRAIN"]["LOADER"].update(samples_per_gpu=4, workers_per_gpu=0)
    config["TEST"]["LOADER"]["workers_per_gpu"] = 0
    path = Path(out_dir) / "pretrain.yaml"
    path.write_text(yaml.safe_dump(config))
    run_script(TOOLS / "source_pretrain" / "main.py", path, "--work-dir=source",
               f"--data-root={data_root}", f"--logs-root={out_dir}",
               f"--backbone-path={backbone_ckpt}")
    return Path(out_dir) / "source" / "model_best.pth"


def _run_script(argv):
    """``run-script`` action: run a training script with .cuda() made a no-op.

    Without GPU, CUDA_VISIBLE_DEVICES="0,1" makes the scripts wrap the models in
    DataParallel, which then runs the module directly on CPU.
    """
    import runpy

    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.cuda.synchronize = lambda *args, **kwargs: None
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


def snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        losses, model = run_client_round(make_market(Path(tmp) / "data"), Path(tmp) / "work")
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return {"torch": torch.__version__, "losses": losses, "weights_sha256": digest.hexdigest()}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "run-script":
        _run_script(sys.argv[2:])
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["save", "compare"])
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    current = snapshot()
    if args.action == "save":
        args.path.write_text(json.dumps(current, indent=1))
        print(f"saved {len(current['losses'])} steps to {args.path}")
        return 0

    reference = json.loads(args.path.read_text())
    same_losses = reference["losses"] == current["losses"]
    same_weights = reference["weights_sha256"] == current["weights_sha256"]
    print("losses identical:", same_losses)
    print("weights identical:", same_weights)
    if reference["torch"] != current["torch"]:
        print(f"warning: saved with torch {reference['torch']}, run with {current['torch']}")
    return 0 if same_losses and same_weights else 1


if __name__ == "__main__":
    sys.exit(main())
