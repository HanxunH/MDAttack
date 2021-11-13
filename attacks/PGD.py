import time
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .utils import adv_check_and_update, one_hot_tensor


def cw_loss(logits, y):
    correct_logit = torch.sum(torch.gather(logits, 1, (y.unsqueeze(1)).long()).squeeze())
    tmp1 = torch.argsort(logits, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    wrong_logit = torch.sum(torch.gather(logits, 1, (new_y.unsqueeze(1)).long()).squeeze())
    loss = - F.relu(correct_logit-wrong_logit)
    return loss


def margin_loss(logits, y):
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(1, (logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss


class PGDAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_restarts=1, v_min=0., v_max=1., num_classes=10,
                 random_start=False, type='PGD', use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.type = type
        self.use_odi = use_odi

    def _get_rand_noise(self, X):
        eps = self.epsilon
        device = X.device
        return torch.FloatTensor(*X.shape).uniform_(-eps, eps).to(device)

    def perturb(self, x_in, y_in):
        model = self.model
        device = x_in.device
        epsilon = self.epsilon
        X_adv = x_in.detach().clone()
        X_pgd = Variable(x_in.data, requires_grad=True)
        nc = torch.zeros_like(y_in)

        for r in range(self.num_restarts):
            if self.random_start:
                random_noise = self._get_rand_noise(x_in)
                X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

            if self.use_odi:
                out = model(x_in)
                rv = torch.FloatTensor(*out.shape).uniform_(-1., 1.).to(device)

            for i in range(self.num_steps):
                opt = optim.SGD([X_pgd], lr=1e-3)
                opt.zero_grad()
                if self.use_odi and i < 2:
                    loss = (model(X_pgd) * rv).sum()
                elif self.use_odi:
                    loss = margin_loss(model(X_pgd), y_in)
                elif self.type == 'CW':
                    loss = cw_loss(model(X_pgd), y_in)
                else:
                    loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y_in)
                loss.backward()
                if self.use_odi and i < 2:
                    eta = epsilon * X_pgd.grad.data.sign()
                else:
                    eta = self.step_size * X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                eta = torch.clamp(X_pgd.data - x_in.data, -epsilon, epsilon)
                X_pgd = Variable(x_in.data + eta, requires_grad=True)
                X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
                logits = self.model(X_pgd)
                X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv


class MTPGDAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_restarts=1, v_min=0., v_max=1., num_classes=10,
                 random_start=False, use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.use_odi = use_odi

    def _get_rand_noise(self, X):
        eps = self.epsilon
        device = X.device
        return torch.FloatTensor(*X.shape).uniform_(-eps, eps).to(device)

    def perturb(self, x_in, y_in):
        model = self.model
        device = x_in.device
        epsilon = self.epsilon
        X_adv = x_in.detach().clone()
        X_pgd = Variable(x_in.data, requires_grad=True)
        nc = torch.zeros_like(y_in)

        for t in range(self.num_classes):
            y_gt = one_hot_tensor(y_in, self.num_classes)
            y_tg = torch.zeros_like(y_in)
            y_tg += t
            y_tg = one_hot_tensor(y_tg, self.num_classes)
            # y_tg = one_hot_tensor(targets, self.num_classes)
            for r in range(self.num_restarts):
                if self.random_start:
                    random_noise = self._get_rand_noise(x_in)
                    X_pgd = Variable(X_pgd.data+random_noise,
                                     requires_grad=True)

                if self.use_odi:
                    out = model(x_in)
                    rv = torch.FloatTensor(*out.shape).uniform_(-1., 1.)
                    rv = rv.to(device)

                for i in range(self.num_steps):
                    opt = optim.SGD([X_pgd], lr=1e-3)
                    opt.zero_grad()
                    if self.use_odi and i < 2:
                        loss = (model(X_pgd) * rv).sum()
                    else:
                        z = model(X_pgd)
                        z_y = y_gt * z
                        z_t = y_tg * z
                        loss = (-z_y+z_t).mean()
                    loss.backward()
                    if self.use_odi and i < 2:
                        eta = epsilon * X_pgd.grad.data.sign()
                    else:
                        eta = self.step_size * X_pgd.grad.data.sign()
                    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                    eta = torch.clamp(X_pgd.data - x_in.data, -epsilon, epsilon)
                    X_pgd = Variable(x_in.data + eta, requires_grad=True)
                    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0),
                                     requires_grad=True)
                    logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv
