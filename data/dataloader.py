import os
import random

import PIL
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Resize


def load_path(path):
    names = os.listdir(path)
    paths = []
    for name in names:
        mask = os.path.join(path, f"{name[:-4]}_mask.png")
        if os.path.exists(mask):
            paths.append((os.path.join(path, name), mask))
    return paths


class LoadImageDataset(Dataset):
    def __init__(self, file_paths, data_root, img_size, apply_transform=False):

        self.file_paths = file_paths
        self.data_root = data_root
        self.apply_transform = apply_transform
        self.resize_inp = Resize((img_size, img_size), interpolation=PIL.Image.NEAREST)
        self.resize_mask = Resize((img_size, img_size), interpolation=PIL.Image.NEAREST)

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def transform(image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def __getitem__(self, idx):
        inp_img_path = self.file_paths[idx][0]
        inp_img = (
            self.resize_inp(
                torchvision.transforms.PILToTensor()(
                    PIL.Image.open(os.path.join(self.data_root, inp_img_path)).convert("RGB")
                )
            )
            / 255.0
        )
        out_img_path = self.file_paths[idx][1]
        out_img = (
            self.resize_mask(
                torchvision.transforms.PILToTensor()(
                    PIL.Image.open(os.path.join(self.data_root, out_img_path)).convert("RGB")
                )
            )[0]
            / 255.0
        )
        if self.apply_transform:
            inp_img, out_img = self.transform(inp_img, out_img)
        sample = (inp_img, out_img)
        return sample


class LoadRoboDataset(Dataset):
    def __init__(self, file_paths, labels, data_root, img_size, apply_transform=False, phase="train"):

        self.file_paths = file_paths
        self.labels = labels
        self.data_root = data_root
        self.apply_transform = apply_transform
        self.resize_inp = Resize((img_size, img_size), interpolation=PIL.Image.NEAREST)
        self.resize_mask = Resize((img_size, img_size), interpolation=PIL.Image.NEAREST)
        self.phase = phase

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def transform(image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def mask_to_class_rgb(self, mask):
        c, h, w = mask.shape
        mask_out = torch.empty(h, w, dtype=torch.long)

        color_to_label = {tuple(v): k for k, v in self.labels.items()}

        for k in color_to_label:
            idx = mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2)
            validx = idx.sum(0) == 3
            mask_out[validx] = torch.tensor(color_to_label[k], dtype=torch.long)

        return mask_out

    def __getitem__(self, idx):
        inp_img_path = self.file_paths[idx][0]
        inp_img = self.resize_inp(
            torchvision.transforms.PILToTensor()(
                PIL.Image.open(os.path.join(self.data_root, self.phase, inp_img_path)).convert("RGB")
            )
            / 255.0
        )
        out_img_path = self.file_paths[idx][1]
        out_img = self.resize_mask(
            torchvision.transforms.PILToTensor()(
                PIL.Image.open(os.path.join(self.data_root, self.phase, out_img_path)).convert("RGB")
            )
        )
        if self.apply_transform:
            inp_img, out_img = self.transform(inp_img, out_img)

        out_img = self.mask_to_class_rgb(out_img)
        sample = (inp_img, out_img)
        return sample
