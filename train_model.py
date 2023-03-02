import argparse
import json
import os

import torch
from torch import nn
from torch.optim import lr_scheduler, Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.ckd import train as ckd_training
from core.ckd_spl import train as ckd_spl_training
from core.train import train as normal_training
from data.dataloader import LoadImageDataset, LoadRoboDataset
from metrics.losses import CurriculumKDLoss
from models.deeplab import DeepLabV3
from models.linknet import LinkNet
from models.unet import UNet
from utils.network_utils import set_seed, remove_dataparallel_wrapper


def arg_parser():
    parser = argparse.ArgumentParser(description="Curriculum KD")

    # Data
    parser.add_argument("--data", default="bus", help="options:[neelde, bus, robo]")
    parser.add_argument("--data-root", default=None, help="path that has the dataset images")
    parser.add_argument("--img-size", default=224, type=int, help="img size")
    parser.add_argument("--batch-size", default=16, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=0.003, type=float, help="initial learning rate")
    parser.add_argument("--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("--model-path", default="outputs/bus/ckd/unet.pth", help="path where model is to be saved")
    parser.add_argument(
        "--teacher-path", default="outputs/bus/unet-reproduce/unet.pth", help="path to pretrained teacher model"
    )
    parser.add_argument("--tb", action="store_true", help="enable tensorboard logging")

    # General Model
    parser.add_argument("--model-name", default="CKD", help="options:[LinkNet, UNet, CKD, KD, DeepLab]")

    # SPL
    parser.add_argument("--spl", action="store_true", help="enable spl while training")
    parser.add_argument("--mu-update", default=0.1, type=float, help="increment mu every epoch-width")
    parser.add_argument("--mu-update-every", default=5, type=int, help="epoch-width")
    parser.add_argument("--initial-mu", default=0.2, type=float, help="Initial threshold value")

    # Loss
    parser.add_argument("--weights-to", default="ce", help="options:[ce, kld, both]")
    parser.add_argument("--temp", default=4, type=float, help="temperature")
    parser.add_argument("--alpha", default=0.3, type=float, help="alpha")
    parser.add_argument(
        "--ckd-loss-type",
        default="ce_weighted",
        choices=[
            "ce_weighted",
            "ce_aliter_weighted",
            "both_weighted_spl_per_px",
            "both_weighted_spl_per_img",
            "both_weighted_spl_per_px_no_alpha",
            "both_weighted_spl_per_px_no_alpha_no_weights",
        ],
    )

    # TS
    parser.add_argument("--confidence-ts", default=6.26, type=float, help="teacher's logits scale to get conf")

    return parser


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.data_root is None:
        args.data_root = os.path.join("datasets", args.data)

    train_paths = json.load(open(os.path.join("datasets", args.data + "_train.json")))
    val_paths = json.load(open(os.path.join("datasets", args.data + "_val.json")))
    apply_transform = True if args.data == "needle" else False
    if args.data == "robo":
        labels_json = json.load(open(os.path.join("datasets", args.data + "_labels.json")))
        labels = {}
        for label_type in labels_json:
            labels[label_type["classid"]] = label_type["color"]
        train_data = LoadRoboDataset(
            train_paths, labels, args.data_root, args.img_size, apply_transform=apply_transform, phase="train"
        )
        val_data = LoadRoboDataset(val_paths, labels, args.data_root, args.img_size, apply_transform=False, phase="val")
    else:
        train_data = LoadImageDataset(train_paths, args.data_root, args.img_size, apply_transform=apply_transform)
        val_data = LoadImageDataset(val_paths, args.data_root, args.img_size, apply_transform=False)

    dataloaders = {
        "train": DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_data, batch_size=args.batch_size, shuffle=True),
    }

    if not os.path.exists(os.path.dirname(args.model_path)):
        os.makedirs(os.path.dirname(args.model_path))

    if args.model_name in ["UNet", "DeepLab", "LinkNet"]:
        if args.model_name == "UNet":
            student_model = UNet(2)
        elif args.model_name == "DeepLab":
            student_model = DeepLabV3(2)
        elif args.model_name == "LinkNet":
            student_model = LinkNet(2)
        else:
            raise Exception
        student_model = student_model.to(device)
        if device == torch.device("cuda"):
            student_model = nn.DataParallel(student_model).to(device)
        loss_func = nn.CrossEntropyLoss()
        training = normal_training

    elif args.model_name in ["CKD", "KD"]:
        teacher_model = UNet(2).to(device)
        student_model = UNet(2).to(device)

        try:
            teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
        except RuntimeError:
            teacher_model.load_state_dict(
                remove_dataparallel_wrapper(torch.load(args.teacher_path, map_location=device))
            )

        teacher_model = nn.DataParallel(teacher_model).to(device)

        weights_to = "kd" if args.model_name == "KD" else args.weights_to
        loss_func = CurriculumKDLoss(weights_to, args.alpha, args.temp, args.confidence_ts)
        training = ckd_spl_training if args.spl else ckd_training
    else:
        raise NotImplementedError

    optimizer = Adam(student_model.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if args.tb:
        writer = SummaryWriter(os.path.join(os.path.dirname(args.model_path), "tb_log"))
    else:
        writer = None

    if args.model_name in ["UNet", "DeepLab", "LinkNet"]:
        student_model, losses = training(
            student_model, dataloaders, loss_func, optimizer, exp_lr_scheduler, num_epochs=args.epochs
        )
    elif args.model_name in ["CKD", "KD"]:
        # noinspection PyUnboundLocalVariable
        student_model, losses = training(
            teacher_model,
            student_model,
            dataloaders,
            loss_func,
            optimizer,
            exp_lr_scheduler,
            num_epochs=args.epochs,
            ckd_loss_type=args.ckd_loss_type,
            initial_mu=args.initial_mu,
            mu_update=args.mu_update,
            mu_update_every=args.mu_update_every,
            writer=writer,
        )
    else:
        raise NotImplementedError

    torch.save(student_model.state_dict(), args.model_path)

    loss_file = open(args.model_path[:-4] + ".json", "w")
    json.dump(losses, loss_file)
    loss_file.close()
    writer.close()


if __name__ == "__main__":
    set_seed(12345)
    parsed = arg_parser().parse_args()
    main(parsed)
