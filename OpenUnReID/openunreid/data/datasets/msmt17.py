# Written by Zhiwei Zhang

import os.path as osp
import warnings

from ..utils.base_dataset import ImageDataset


class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person
            Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """

    dataset_dir = "msmt17"
    dataset_url = None

    def __init__(self, root, mode, n_tasks, task_id, filtre , del_labels=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.n_tasks = n_tasks
        self.task_id = task_id
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, "MSMT17_V1")
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                "The current data structure is deprecated. Please "
                'put data folders such as "train" under '
                '"MSMT17_V1".'
            )

        self.list_path = osp.join(self.dataset_dir, "list_" + mode + ".txt")

        if (mode == "trainval") and (not osp.exists(self.list_path)):
            self.list_train_path = osp.join(self.dataset_dir, "list_train.txt")
            self.list_val_path = osp.join(self.dataset_dir, "list_val.txt")
            self.check_before_run(
                [self.dataset_dir, self.list_train_path, self.list_val_path]
            )
            self.merge_list(self.list_train_path, self.list_val_path, self.list_path)

        subsets_cfgs = {
            "train": (osp.join(self.dataset_dir, "train"), self.list_path),
            "val": (osp.join(self.dataset_dir, "train"), self.list_path),
            "trainval": (osp.join(self.dataset_dir, "train"), self.list_path),
            "query": (osp.join(self.dataset_dir, "test"), self.list_path),
            "gallery": (osp.join(self.dataset_dir, "test"), self.list_path),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(self.mode)
            )

        required_files = [cfgs[0], cfgs[1]]
        self.check_before_run(required_files)

        data = self.process_dir(mode, filtre, *cfgs)
        super(MSMT17, self).__init__(data, mode, **kwargs)

    def process_dir(self, mode, filtre, dir_path, list_path):
        with open(list_path, "r") as txt:
            lines = txt.readlines()
        # get all identities
        if type(filtre)==type(True):
            mode_selection='id'
        elif type(filtre[0])==type("test"):
            mode_selection='path'
        else:
            mode_selection='id'
        pid_container = set()
        for img_info in lines:
            img_path, pid = img_info.split(" ")
            # print(img_path,filtre)
            pid = int(pid)  # no need to relabel
            # if mode == 'trainval' and filtre==True:
            #     if pid%self.n_tasks==self.task_id:
            #         pid_container.add(pid)
            # elif mode == 'trainval' and isinstance(filtre,list) and mode_selection=='id':
            #     # print("enter filtrage with list of similar indices from source")
            #     # print("n_tasks : {}".format(self.n_tasks))
            #     # print("task_id : {}".format(self.task_id))
            #     if pid in filtre:
            #         pid_container.add(pid)
            # elif mode == 'trainval' and isinstance(filtre,list) and mode_selection=='path':
            #     # print("enter filtrage with list of similar paths from source")
            #     img_path_ = '/data/cvlab/home/ramih/Knowledge_Distilation/data/OpenUnReID/datasets/msmt17/MSMT17_V1/train/'+img_path
            #     if img_path_ in filtre:
            #         pid_container.add(pid)

            # else:
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        data = []

        for img_info in lines:
            img_path, pid = img_info.split(" ")
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split("_")[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            if pid in pid_container:
                if filtre==True:
                    if not self.del_labels:
                        if camid==self.task_id:
                            data.append((img_path, pid, camid))
                    else:
                        # use 0 as labels for all images
                        if camid==self.task_id:
                            data.append((img_path, 0, camid))
                else:
                    if not self.del_labels:
                        data.append((img_path, pid, camid))
                    else:
                        # use 0 as labels for all images
                        data.append((img_path, 0, camid))
            else:
                continue

        return data

    def merge_list(self, src1_path, src2_path, dst_path):
        src1 = open(src1_path, "r")
        src2 = open(src2_path, "r")
        dst = open(dst_path, "w")

        for line in src1.readlines():
            dst.write(line.strip() + "\n")
        src1.close()

        for line in src2.readlines():
            dst.write(line.strip() + "\n")
        src2.close()

        dst.close()
