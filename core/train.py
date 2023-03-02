import copy
import time

import torch


def train(model, dataloaders, loss_func, optimizer, scheduler, num_epochs=20):
    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    all_losses = {"train": [], "val": []}
    for epoch in range(num_epochs):
        print("Epoch No. --> {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels.long())

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            all_losses[phase].append(epoch_loss)
            print("{} Loss: {:.4f}".format(phase, epoch_loss))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_taken = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_taken // 60, time_taken % 60))
    print("Best val Loss: {:4f}".format(best_loss))

    model.load_state_dict(best_model_wts)
    return model, all_losses
