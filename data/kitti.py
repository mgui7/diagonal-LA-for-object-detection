import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
import cv2


KITTI_CLASSES = (
'Car',
'Van',
'Truck',
'Pedestrian',
'Person_sitting',
'Cyclist',
'Tram',
'Misc')

KITTI_ROOT = osp.join("data/kitti")


class KittiDetection(Dataset):

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    train = True
    img_files = []
    label_files = []


    def __init__(self, root, img_size=300,transform=None):
        
        self._location = "training" if self.train else "testing"
        
        # self._raw_folder = os.path.join(root, "Kitti" , "raw")
        self._raw_folder = os.path.join(root, "kitti")
        image_dir = os.path.join(self._raw_folder,self._location,self.image_dir_name)
        label_dir = os.path.join(self._raw_folder,self._location,self.labels_dir_name)

        # with open(root, 'r') as file:
        #     self.img_files = file.readlines()
        # self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        
        # Reading image and label filenames
        for img_file in os.listdir(image_dir):
            # Reassuring image files
            if img_file.endswith('png'):
                self.img_files.append( os.path.join(image_dir, img_file) )
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]

        self.img_shape = (img_size, img_size)
        self.max_objects = 50
        self.transform = transform
        self.name = 'kitti'

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt


    def pull_image(self, index):
        img_name = self.img_files[index].rstrip()
        return cv2.imread(img_name, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = np.loadtxt(label_path).reshape(-1, 5)
        return label_path, labels


    def pull_item(self, index):
        #---------
        #  Image
        #---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        # Handles images with less than three channels
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))

        h, w, _ = img.shape
        input_img = img
        padded_h, padded_w, _ = input_img.shape

        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)

            # Calculate ratios from coordinates
            labels[:, 1] = x1 / padded_w
            labels[:, 2] = y1 / padded_h
            labels[:, 3] = x2 / padded_w
            labels[:, 4] = y2 / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        target = [np.append(a[1:],a[0]) for a in labels]

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), target, h, w

    def __len__(self):
        return len(self.img_files)
