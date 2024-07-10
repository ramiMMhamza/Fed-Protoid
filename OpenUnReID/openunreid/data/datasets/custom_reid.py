# Written by Zhiwei Zhang

import os.path as osp
import warnings
import pandas as pd
from ..utils.base_dataset import ImageDataset


class CustomDataset(ImageDataset):
    """This class allows to load data from a csv file.
    The csv file consists of the three following columns ["path", "id", "camid"] 
    and must be named as list_mode.csv with mode = train, val, trainval, query, or gallery
    """

    dataset_dir = "randperson_subset_train"
    dataset_url = None

    def __init__(self, root, mode, filtre=None, del_labels=False, subdir="", **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels

        dataset_dir = osp.join(self.dataset_dir, subdir)
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                "The current data structure is deprecated. Please "
                'put data folders such as "train" under {}'.format(subdir)
            )

        subsets_cfgs = ["train", "val", "trainval", "query", "gallery"]
        if mode not in subsets_cfgs:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(mode)
            )

        self.list_path = osp.join(self.dataset_dir, "list_" + mode + ".csv")
        self.check_before_run([self.list_path])

        data = self.process_dir(self.list_path, filtre=filtre)
        self.__class__.__name__ = subdir
        super(CustomDataset, self).__init__(data, mode, **kwargs)

    def process_dir(self, data_path, filtre=None):
        df = pd.read_csv(data_path)
        cols = ["path", "id", "camid"]
        df["id"] = df["id"].astype(int)
        df["camid"] = df["camid"].astype(int)
        if isinstance(filtre,list):
            print("enter filtration similarities for randperson")
            df[df["id"].isin(filtre)]
        if self.del_labels:
            df["id"] = 0
        # convert dataframe to the list of tuples
        data = list(df[cols].itertuples(index=False, name=None))

        return data

