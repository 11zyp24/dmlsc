from torch.utils.data import Dataset
import torch
import json, os, random, time
import cv2

def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list


class BaseSet(Dataset):
    def __init__(self, mode, json_path, INPUT_SIZE=(32, 32), COLOR_SPACE='RGB', transform=None):
        self.mode = mode
        self.transform = transform
        self.input_size = INPUT_SIZE
        self.color_space = COLOR_SPACE
        self.size = self.input_size

        print("Use {} Mode to train network".format(self.color_space))


        if self.mode == "train":
            print("Loading train data ...", end=" ")
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
        else:
            raise NotImplementedError
        self.json_path = json_path
        self.update_transform()

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        self.num_classes = self.all_info["num_classes"]

        self.data = self.all_info['annotations']

        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))


    def update(self, epoch):
        self.epoch = epoch
        # if self.sample_type == "weighted_progressive":
            # self.progress_p = epoch / self.cfg.TRAIN.MAX_EPOCH * self.class_p + (
            #             1 - epoch / self.cfg.TRAIN.MAX_EPOCH) * self.instance_p
            # print('self.progress_p', self.progress_p)


    def __getitem__(self, index):
        print('start get item...')
        now_info = self.data[index]
        img = self._get_image(now_info)
        print('complete get img...')
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info["category_id"] if "test" not in self.mode else 0
        )  # 0-index
        if self.mode not in ["train", "valid"]:
           meta["image_id"] = now_info["image_id"]
           meta["fpath"] = now_info["fpath"]

        return image, image_label, meta

    def update_transform(self, input_size=None):
        # normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        # transform_list = [transforms.ToPILImage()]
        # transform_ops = (
        #     self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
        #     if self.mode == "train"
        #     else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        # )
        # for tran in transform_ops:
        #     transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        # transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = None

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.all_info['annotations']

    def __len__(self):
        return len(self.all_info['annotations'])

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "pillow open {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_trans_image(self, img_idx):
        now_info = self.data[img_idx]
        fpath = os.path.join(now_info["fpath"])
        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)[None, :, :, :]

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i if i != 0 else 0 for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

