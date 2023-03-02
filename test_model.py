import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from core.test import test
from data.dataloader import LoadImageDataset
from models.deeplab import DeepLabV3
from models.linknet import LinkNet
from models.unet import UNet
from utils.network_utils import set_seed, remove_dataparallel_wrapper


def arg_parser():
    parser = argparse.ArgumentParser(description="Curriculum KD")
    parser.add_argument("--data", default="bus", help="options:[needle, bus]")
    parser.add_argument("--data-root", default=None, help="path that has the dataset images")
    parser.add_argument("--img-size", default=224, type=int, help="img size")
    parser.add_argument("--batch-size", default=16, type=int, help="mini-batch size")
    parser.add_argument("--model-name", default="UNet", help="options: [LinkNet, UNet, CKD, KD, DeepLab]")
    parser.add_argument("--model-path", default="outputs/bus/unet/unet.pth", help="Path to trained model weights")
    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_root is None:
        args.data_root = os.path.join("datasets", args.data)
    val_paths = json.load(open(os.path.join("datasets", args.data + "_val.json")))

    val_data = LoadImageDataset(val_paths, args.data_root, args.img_size, apply_transform=False)
    dataloaders = {"val": DataLoader(val_data, batch_size=args.batch_size, shuffle=True)}

    if args.model_name == "DeepLab":
        seg_model = DeepLabV3(2)
    elif args.model_name == "LinkNet":
        seg_model = LinkNet(2)
    else:
        seg_model = UNet(2)

    seg_model = seg_model.to(device)
    try:
        seg_model.load_state_dict(torch.load(args.model_path, map_location=device))
    except RuntimeError:
        seg_model.load_state_dict(remove_dataparallel_wrapper(torch.load(args.model_path, map_location=device)))
    if device == torch.device("cuda"):
        seg_model = nn.DataParallel(seg_model).to(device)

    metrics = test(seg_model, dataloaders)
    print("Copy paste CSV")
    print(",".join(metrics.keys()))
    print(",".join([str(np.round(data, 4)) for data in metrics.values()]))


if __name__ == "__main__":
    set_seed(12345)
    parsed = arg_parser().parse_args()
    main(parsed)
