# Written by Yixiao Ge

import glob
import os.path as osp
import re
import warnings

from ..utils.base_dataset import ImageDataset


class CUHK03(ImageDataset):
    """CUHK03.
    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """

    dataset_dir = "cuhk03"
    dataset_url = "http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip"

    def __init__(self, root, mode, n_tasks, task_id, filtre , val_split=0.2, del_labels=False, **kwargs):
        self.mode = mode
        self.n_tasks = n_tasks
        self.task_id = task_id
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        # self.download_dataset(self.dataset_dir, self.dataset_url)
        assert (val_split > 0.0) and (
            val_split < 1.0
        ), "the percentage of val_set should be within (0.0,1.0)"

        # allow alternative directory structure
        # dataset_dir = osp.join(self.dataset_dir, "Market-1501-v15.09.15")
        # if osp.isdir(dataset_dir):
        #     self.dataset_dir = dataset_dir
        # else:
        #     warnings.warn(
        #         "The current data structure is deprecated. Please "
        #         'put data folders such as "bounding_box_train" under '
        #         '"Market-1501-v15.09.15".'
        #     )

        subsets_cfgs = {
            "train": (
                osp.join(self.dataset_dir, "cuhk03_x-100_TRAIN"),
                [0.0, 1.0 - val_split],
                True,
            ),
            "val": (
                osp.join(self.dataset_dir, "cuhk03_x-100_TRAIN"),
                [1.0 - val_split, 1.0],
                False,
            ),
            "trainval": (
                osp.join(self.dataset_dir, "cuhk03_x-100_TRAIN"),
                [0.0, 1.0],
                True,
            ),
            "query": (osp.join(self.dataset_dir, "cuhk03_x-100_TEST"), [0.0, 1.0], False), #tmp
            "gallery": (
                osp.join(self.dataset_dir, "cuhk03_x-100_TEST"),
                [0.0, 1.0 - val_split],
                False,
            ),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(self.mode)
            )

        required_files = [self.dataset_dir, cfgs[0]]
        self.check_before_run(required_files)

        data = self.process_dir(mode, filtre, *cfgs)
        super(CUHK03, self).__init__(data, mode, **kwargs)

    def process_dir(self, mode, filtre, dir_path, data_range, relabel=False):
        ids_paths = glob.glob(osp.join(dir_path, "*"))
        img_paths = []
        for i in range (len(ids_paths)):
            img_paths += glob.glob(osp.join(ids_paths[i],"*.jpg")) 
        pattern = re.compile(r"([-\d]+)_(\d)")

        # get all identities
        pid_container = set()
        for img_path in img_paths:
            (pid_,_) = pattern.search(img_path).groups()
            cid = int(str(pid_)[0])
            pid = int(str(pid_)[1:])
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid_container = sorted(pid_container)

        # select a range of identities (for splitting train and val)
        start_id = int(round(len(pid_container) * data_range[0]))
        end_id = int(round(len(pid_container) * data_range[1]))
        pid_container = pid_container[start_id:end_id]
        assert len(pid_container) > 0
       
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if mode == 'trainval' and filtre==True:
            print("enter filtrage for market")
            print("n_tasks : {}".format(self.n_tasks))
            print("task_id : {}".format(self.task_id))
            pid_container_bis = set()
            for key,value in pid2label.items():
                # if self.task_id==0:
                #     if key%self.n_tasks==self.task_id+1:
                #         pid_container_bis.add(key)
                # if self.task_id==1:
                #     if key%self.n_tasks==self.task_id-1:
                #         pid_container_bis.add(key)
                # else:
                if key%self.n_tasks==self.task_id:
                    pid_container_bis.add(key)
            pid_container = pid_container_bis
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
        elif mode == 'trainval' and isinstance(filtre,list):
            print("enter filtrage with list indices")

            pid_container_bis = set()
            for key,value in pid2label.items():
                for task_id_ in filtre:
                    if key%self.n_tasks==task_id_:
                        pid_container_bis.add(key)
                # if key in filtre:
                #     pid_container_bis.add(key)
            pid_container = pid_container_bis
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
        data = []
        #if self.mode =='trainval':
        #    img_paths = img_paths[9000:]
        for img_path in img_paths:
            (pid_,_) = pattern.search(img_path).groups()
            camid = int(str(pid_)[0])
            pid = int(str(pid_)[1:])
            if (pid not in pid_container) or (pid == -1):
                continue

            assert 0 <= pid <= 1467 #1501  # pid == 0 means background
            assert 1 <= camid <= 3
            camid -= 1  # index starts from 0

            if not self.del_labels:
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))
            else:
                # use 0 as labels for all images
                data.append((img_path, 0, camid))
                # data.append((img_path, pid, camid))
        return data
