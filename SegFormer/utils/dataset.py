import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ChangeDataset(Dataset):
    def __init__(
        self,
        root_dir,
        img_size=256,
        return_name=False,
        num_classes=7,
        remap_labels=True,
        label_values=None,
    ):
        self.root = root_dir
        self.img_size = img_size
        self.return_name = return_name
        self.num_classes = num_classes
        self.remap_labels = remap_labels
        self.label_values = label_values

        im1_dir = os.path.join(root_dir, "im1")
        if not os.path.isdir(im1_dir):
            raise FileNotFoundError(f"Missing im1 directory: {im1_dir}")

        self.files = sorted(os.listdir(im1_dir))

        self.label_lut = None
        if self.remap_labels:
            self.label_lut = self._build_label_lut()

    def _build_label_lut(self):
        if self.label_values is not None:
            if len(self.label_values) != self.num_classes:
                raise ValueError(
                    f"label_values length ({len(self.label_values)}) must match num_classes ({self.num_classes})."
                )
            lut = np.zeros(256, dtype=np.uint8)
            for idx, val in enumerate(self.label_values):
                lut[int(val)] = idx
            return lut

        labels_dir = os.path.join(self.root, "labels")
        values = set()

        for name in self.files:
            p_label = os.path.join(labels_dir, name)
            label = cv2.imread(p_label, 0)
            if label is None:
                continue
            values.update(np.unique(label).tolist())

        values = sorted(values)
        if len(values) == 0:
            return None

        if max(values) < self.num_classes and values == list(range(len(values))):
            return None

        if len(values) > self.num_classes:
            raise ValueError(
                f"Found {len(values)} unique label values, but num_classes={self.num_classes}."
            )

        lut = np.zeros(256, dtype=np.uint8)
        for idx, val in enumerate(values):
            lut[int(val)] = idx

        return lut

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        p_im1 = os.path.join(self.root, "im1", name)
        p_im2 = os.path.join(self.root, "im2", name)
        p_label = os.path.join(self.root, "labels", name)

        im1 = cv2.imread(p_im1)
        im2 = cv2.imread(p_im2)
        label = cv2.imread(p_label, 0)
        if label is None:
            raise FileNotFoundError(f"Label not found or unreadable: {p_label}")

        if self.label_lut is not None:
            label = self.label_lut[label]

        im1 = cv2.resize(im1, (self.img_size, self.img_size))
        im2 = cv2.resize(im2, (self.img_size, self.img_size))
        label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        im1 = im1 / 255.0
        im2 = im2 / 255.0

        img = np.concatenate([im1, im2], axis=2)

        img = torch.tensor(img).permute(2, 0, 1).float()
        label = torch.tensor(label).long()

        if self.return_name:
            return img, label, name

        return img, label