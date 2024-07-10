from .cuhk03np import CUHK03NP
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .vehicleid import VehicleID
from .vehiclex import VehicleX
from .veri import VeRi
from .cuhk03 import CUHK03
from .custom_reid import CustomDataset
__all__ = ["build_dataset", "names"]

__factory = {
    "custom": CustomDataset,
    "cuhk03-np": CUHK03NP,
    "cuhk03": CUHK03,
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "msmt17": MSMT17,
    "personx": PersonX,
    "veri": VeRi,
    "vehicleid": VehicleID,
    "vehiclex": VehicleX,
}


def names():
    return sorted(__factory.keys())


def build_dataset(name, root, mode, n_tasks, task_id,filtre, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    mode : str
        The subset for the dataset, e.g. [train | val | trainval | query | gallery]
    val_split : float, optional
        The proportion of validation to all the trainval. Default: 0.3
    del_labels: bool, optional
        If true, delete all ground-truth labels and replace them with all zeros.
        Default: False
    transform : optional
        The transform for dataloader. Default: None
    """
    if "custom_" == name[:7]:
        data_name = name[7:]
        print("Loading", data_name, "dataset")
        return __factory["custom"](root, mode, filtre, *args, subdir="", **kwargs)
    elif name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, mode, n_tasks, task_id,filtre, *args, **kwargs)
