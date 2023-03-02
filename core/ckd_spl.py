import copy
import time

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from core.test import test


def train(
    teacher_model,
    student_model,
    dataloaders,
    loss_func,
    optimizer,
    scheduler,
    num_epochs=20,
    ckd_loss_type="ce_weighted",
    initial_mu=0.2,
    mu_update=0.1,
    mu_update_every=5,
    writer: SummaryWriter = None,
):
    start_time = time.time()
    mu = initial_mu
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_loss = 50.0
    all_losses = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1, 1):
        print("Epoch No. --> {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        print(mu)

        if writer is not None:
            writer.add_scalar("ckd/mu", mu, epoch)

        teacher_model.eval()

        num_ones = 0
        num_zeros = 0
        for inputs, _ in dataloaders["train"]:
            inputs = inputs.cuda()
            with torch.no_grad():
                teacher_out = teacher_model(inputs)
            out_confidence = F.softmax(teacher_out / loss_func.confidence_ts, dim=1)
            weights = torch.where(1 - out_confidence <= mu, 1.0, 0.0)  # per pixel wise loss
            weights = torch.sum(weights, dim=1).clamp(
                min=0, max=1
            )  # sum over the images channels of one image of a batch
            num_zeros += torch.logical_not(weights).sum().item()
            num_ones += weights.sum().item()

        if writer is not None:
            writer.add_scalar("spl/data_in_train", num_ones / (num_ones + num_zeros), epoch)

        for phase in ["train", "val"]:
            if phase == "train":
                student_model.train()
            else:
                student_model.eval()
            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_out = teacher_model(inputs)
                with torch.set_grad_enabled(phase == "train"):
                    outputs = student_model(inputs)
                    loss = getattr(loss_func, ckd_loss_type)(outputs, teacher_out, labels, mu)
                    eval_loss = loss_func.ce_loss(outputs, labels).mean()
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += eval_loss.item() * inputs.size(0)
            if phase == "train":
                scheduler.step()
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_losses[phase].append(epoch_loss)
            print("{} Loss: {:.4f}".format(phase, epoch_loss))
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(student_model.state_dict())

            if phase == "val":
                metrics = test(student_model, dataloaders)

            if writer is not None:
                writer.add_scalar(f"logs/{phase}_loss", epoch_loss, epoch)
                if phase == "val":
                    for name, value in metrics.items():
                        writer.add_scalar(f"logs/metric_{name}", value, epoch)

        if writer is not None:
            writer.add_scalar("logs/lr", scheduler.optimizer.param_groups[0]["lr"], epoch)

        if not epoch % mu_update_every:  # Update regularly
            mu += mu_update
            mu = min(mu, 1)

    time_taken = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_taken // 60, time_taken % 60))
    print("Best val Loss: {:4f}".format(best_loss))
    student_model.load_state_dict(best_model_wts)

    return student_model, all_losses
