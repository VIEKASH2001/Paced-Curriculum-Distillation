import copy
import time

import torch


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
    **kwargs,
):
    start_time = time.time()
    mu = initial_mu
    best_model_wts = copy.deepcopy(student_model.state_dict())
    best_loss = 50.0
    all_losses = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print("Epoch No. --> {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        print(mu)

        teacher_model.eval()
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
        if not epoch % 5:  # UPDATION
            mu += 0.1
            mu = min(mu, 1)
        # mu = mu*1.2

    time_taken = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_taken // 60, time_taken % 60))
    print("Best val Loss: {:4f}".format(best_loss))
    student_model.load_state_dict(best_model_wts)

    return student_model, all_losses
