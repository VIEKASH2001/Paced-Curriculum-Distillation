import torch
import torch.nn.functional as F
from torch import nn


class CurriculumKDLoss:
    def __init__(self, weights_to="kd", alpha=0.5, temp=3, confidence_ts=1.0):
        self.weights_to = weights_to
        self.alpha = alpha
        self.temp = temp

        # self.cda_temp = torch.Tensor([[[temp + (0.923)/100]],[[temp + (0.077)/100]]]).cuda()
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.kl = nn.KLDivLoss(reduction="none")
        print("CurriculumKD Loss", weights_to, alpha, self.temp)
        # print("CurriculumKD_Loss", weights_to, alpha, self.cda_temp, self.temp)
        if self.weights_to == "kd":
            self.value = self.kd
        if self.weights_to == "curriculum":
            self.value = self.curriculum
        if self.weights_to == "ce":
            self.value = self.ce_weighted
        if self.weights_to == "kld":
            self.value = self.kld_weighted
        if self.weights_to == "both":
            self.value = self.both_weighted

        self.confidence_ts = confidence_ts

    def get_weights(self, teacher_out, labels, mu):
        out_confidence = F.softmax(teacher_out / self.confidence_ts, dim=1)
        weights = torch.ones(labels.shape, dtype=torch.float64).cuda() * (min(1, mu + 0.1))
        weights[torch.logical_and(out_confidence[:, 1, :, :] >= 1 - mu, labels == 1)] = 1
        weights[torch.logical_and(out_confidence[:, 0, :, :] >= 1 - mu, labels == 0)] = 1
        return weights

    def get_weights_spl_per_px(self, teacher_out, labels, mu):  # SPL
        out_confidence = F.softmax(teacher_out / self.confidence_ts, dim=1)
        weights = torch.where(1 - out_confidence <= mu, 1.0, 0.0)  # per pixel wise loss
        weights = torch.sum(weights, dim=1).clamp(min=0, max=1)  # sum over the images channels of one image of a batch
        return weights

    def get_weights_spl_per_img(self, teacher_out, labels, mu):  # SPL
        out_confidence = F.softmax(teacher_out / self.confidence_ts, dim=1)
        weights = torch.where(1 - torch.mean(out_confidence, dim=1) <= mu, 1.0, 0.0)  # per avg image loss
        # weights = torch.sum(weights,dim = 1).clamp(min=0, max=1) #sum over the images channels of one image of a batch
        return weights

    def ce_loss(self, outputs, labels):
        return self.ce(outputs, labels.long())

    def kld_loss(self, teacher_out, outputs):
        # teacher_out_temp = F.softmax(teacher_out / self.cda_temp, dim=1)
        # outputs_temp = F.log_softmax(outputs/self.cda_temp, dim=1)
        # kl = self.kl(outputs_temp, teacher_out_temp)*self.cda_temp[0][0][0]*self.cda_temp[0][0][0]

        teacher_out_temp = F.softmax(teacher_out / self.temp, dim=1)
        outputs_temp = F.log_softmax(outputs / self.temp, dim=1)
        kl = self.kl(outputs_temp, teacher_out_temp) * self.temp * self.temp

        kl = torch.mean(kl, 1)
        return kl

    def curriculum(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights(teacher_out, labels, mu) * self.ce_loss(outputs, labels)
        return loss.mean()

    def kd(self, outputs, teacher_out, labels, mu):
        loss = self.alpha * self.kld_loss(teacher_out, outputs) + (1 - self.alpha) * self.ce_loss(outputs, labels)
        return loss.mean()

    def ce_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.alpha * self.kld_loss(teacher_out, outputs) + (1 - self.alpha) * self.get_weights(
            teacher_out, labels, mu
        ) * self.ce_loss(outputs, labels)
        return loss.mean()

    def kld_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.alpha * self.get_weights(teacher_out, labels, mu) * self.kld_loss(teacher_out, outputs) + (
            1 - self.alpha
        ) * self.ce_loss(outputs, labels)
        return loss.mean()

    def both_weighted(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights(teacher_out, labels, mu) * (
            self.alpha * self.kld_loss(teacher_out, outputs) + (1 - self.alpha) * self.ce_loss(outputs, labels)
        )
        return loss.mean()

    def both_weighted_spl_per_px(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights_spl_per_px(teacher_out, labels, mu) * (
            self.alpha * self.kld_loss(teacher_out, outputs) + (1 - self.alpha) * self.ce_loss(outputs, labels)
        )
        return loss.mean()

    def both_weighted_spl_per_px_no_alpha(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights_spl_per_px(teacher_out, labels, mu) * (
            self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels)
        )
        return loss.mean()

    def both_weighted_spl_per_px_no_alpha_no_weights(self, outputs, teacher_out, labels, mu):
        loss = self.kld_loss(teacher_out, outputs) + self.ce_loss(outputs, labels)
        return loss.mean()

    def both_weighted_spl_per_img(self, outputs, teacher_out, labels, mu):
        loss = self.get_weights_spl_per_img(teacher_out, labels, mu) * (
            self.alpha * self.kld_loss(teacher_out, outputs) + (1 - self.alpha) * self.ce_loss(outputs, labels)
        )
        return loss.mean()

    def ce_aliter_weighted(self, outputs, teacher_out, labels, mu):
        weights = self.get_weights(teacher_out, labels, mu)
        loss = (1 - weights * self.alpha) * self.ce_loss(outputs, labels) + weights * self.alpha * self.kld_loss(
            teacher_out, outputs
        )
        return loss.mean()
