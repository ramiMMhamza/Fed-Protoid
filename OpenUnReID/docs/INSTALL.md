## Installation

### Requirements

+ Linux
+ Python 3.10 or newer
+ PyTorch 1.10 or higher, with CUDA (the training scripts only run on GPU)
+ faiss (`faiss-gpu`, or `faiss-cpu` together with `search_type: 3` in the config)

The code base was developed with Python 3.10. The package installation and the
CPU-only parts (data pipeline, clustering, evaluation metric) were also checked
with Python 3.13, PyTorch 2.15 and NumPy 2.5.

### Install

**a.** Create a conda virtual environment and activate it.
```shell
conda create -n fedprotoid python=3.10 -y
conda activate fedprotoid
```

**b.** Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

**c.** Install faiss.
```shell
conda install -c pytorch faiss-gpu   # or: pip install faiss-cpu
```

**d.** Install the dependent libraries and the `openunreid` package (run from the repository root).
```shell
pip install -r OpenUnReID/requirements.txt
pip install -e OpenUnReID
```


### Prepare datasets

Datasets are read from `DATA_ROOT` (`./data/datasets/` by default, relative to the repository root). It is recommended to symlink your dataset root there, or to pass `--data-root` to the training scripts.

Download the datasets from
+ DukeMTMC-reID: [[Google Drive]](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view) [[Baidu Pan]](https://pan.baidu.com/share/init?surl=jS0XM7Var5nQGcbf9xUztw) (password: bhbh)
+ Market-1501-v15.09.15: [[Google Drive]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu Pan]](https://pan.baidu.com/s/1ntIi2Op)
+ subset1 (PersonX): [[Google Drive]](https://drive.google.com/file/d/1hiHoDt3u7_GfeICMdEBt2Of8vXr1RF-U/view)
+ MSMT17_V1: [[Home Page]](https://www.pkuvmc.com/dataset.html) (request link by email the holder)
+ VehicleID_V1.0: [[Home Page]](https://www.pkuml.org/resources/pku-vehicleid.html) (request link by email the holder)
+ AIC20_ReID_Simulation (VehicleX): [[Home Page]](https://www.aicitychallenge.org/2020-track2-download/) (request password by email the holder)
+ VeRi_with_plate: [[Home Page]](https://github.com/JDAI-CV/VeRidataset#2-download) (request link by email the holder)

Save them under
```shell
data
└── datasets
    ├── dukemtmcreid
    │   └── DukeMTMC-reID
    ├── market1501
    │   └── Market-1501-v15.09.15
    ├── msmt17
    │   └── MSMT17_V1
    ├── personx
    │   └── subset1
    ├── vehicleid
    │   └── VehicleID_V1.0
    ├── vehiclex
    │   └── AIC20_ReID_Simulation
    └── veri
        └── VeRi_with_plate
```

<!-- ### Prepare pre-trained weights

If you want to use [ResNet-IBN](https://arxiv.org/abs/1807.09441) as the backbone, which may perform better than plain ResNet, you need to download the ImageNet pre-trained weights manually from [resnet50_ibn_a](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth), [resnet101_ibn_a](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth). And put them under the folder of `~/.cache/torch/checkpoints/` like
```shell
~/.cache/torch
└── checkpoints
    ├── resnet50_ibn_a-d9d0bb7b.pth # manually downloaded and saved here
    ├── resnet101_ibn_a-59ea0ac6 # manually downloaded and saved here
    └── resnet50-19c8e357.pth # automatically downloaded by python script
``` -->
