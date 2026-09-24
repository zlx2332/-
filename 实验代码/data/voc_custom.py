import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
from data.cls_to_names import voc2007_classes

class VOCCustom(Dataset):
    def __init__(self, split_name, dataset_dir, transform):
        """
        split_name: 'train_split', 'val_split', 'test_split'
        """
        self.dataset_dir = os.path.join(dataset_dir, "VOCdevkit", "VOC2007")
        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")
        self.voc2007_classes = voc2007_classes
        
        # 读取划分名单
        split_file = os.path.join(self.dataset_dir, 'ImageSets/Main', f'{split_name}.txt')
        self.im_name_list = []
        with open(split_file, 'r') as f:
            for line in f:
                self.im_name_list.append(line.strip())
                
        # 严格读取这些图片的多标签真值 (Ground-Truth)
        self.labels = self.read_object_labels(self.dataset_dir)
        
        self.test = []
        for name in self.im_name_list:
            img_path = os.path.join(self.image_dir, f"{name}.jpg")
            label = self.labels[name]
            label = list(set(label))
            if label:
                self.test.append([img_path, label])
        self.transform = transform
        
    def read_object_labels(self, path):
        path_labels = os.path.join(path, 'ImageSets', 'Main')
        labeled_data = defaultdict(list)
        num_classes = len(self.voc2007_classes)

        # 这里的原始txt标注是全集的，我们只提取我们名单里对应的部分
        for i in range(num_classes):
            # trainval 和 test 是官方原始文件，我们需要在两者中共同检索
            for phase in ['trainval', 'test']:
                file = os.path.join(path_labels, f"{self.voc2007_classes[i]}_{phase}.txt")
                if not os.path.exists(file):
                    continue
                with open(file, 'r') as f:
                    for line in f:
                        tmp = line.strip().split()
                        name = tmp[0]
                        label = int(tmp[-1])
                        if label == 1:
                            labeled_data[name].append(i)
        return labeled_data

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        img_path, label = self.test[idx]
        image = Image.open(open(img_path, "rb")).convert("RGB")
        image = self.transform(image)
        
        # 将多标签表示为一个 20 维的 Multi-hot 0/1 向量，方便传统模型计算 BCE 损失
        target = torch.zeros(len(self.voc2007_classes), dtype=torch.float32)
        for l in label:
            target[l] = 1.0
        return image, img_path, target