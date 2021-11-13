import time
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .utils import adv_check_and_update, one_hot_tensor


class MDAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_random_starts=1, v_min=0., v_max=1., change_point=25,
                 first_step_size=16./255., seed=0, norm='Linf', num_classes=10,
                 use_odi=False, use_dlr=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_random_starts = num_random_starts
        self.v_min = v_min
        self.v_max = v_max
        self.change_point = change_point
        self.first_step_size = first_step_size
        self.seed = seed
        self.norm = norm
        self.num_classes = num_classes
        self.use_odi = use_odi
        self.use_dlr = use_dlr
        self.initial_step_size = 2.0 * epsilon

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted.sum(dim=1) + 1e-12)

    def perturb(self, x_in, y_in):
        change_point = self.change_point
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        nat_logits = self.model(x)
        nat_pred = nat_logits.max(dim=1)[1]
        nat_correct = (nat_pred == y).squeeze()

        y_gt = F.one_hot(y, self.num_classes)

        nc = torch.zeros_like(y)
        X_adv = x.detach().clone()
        gt_prev = 0

        for _ in range(max(self.num_random_starts, 1)):
            for r in range(2):
                r_noise = torch.FloatTensor(*x.shape).uniform_(-self.epsilon, self.epsilon).cuda()
                X_pgd = Variable(x.data+r_noise, requires_grad=True)
                if self.use_odi:
                    rv = torch.FloatTensor(*nat_logits.shape).uniform_(-1., 1.)
                    rv = rv.to(nat_logits.device)

                for i in range(self.num_steps):
                    X_pgd.requires_grad_()
                    self.model.zero_grad()
                    with torch.enable_grad():
                        logits = self.model(X_pgd)
                        z_max, max_idx = torch.max(logits * (1 - y_gt) - y_gt * 1.e8, dim=1)
                        z_y, _ = torch.max(logits * y_gt - (1 - y_gt) * 1.e8, dim=1)

                        if self.use_odi and i < 2:
                            loss = (logits * rv).sum()
                        elif i < 1:
                            loss_per_sample = z_y
                            loss = torch.mean(loss_per_sample)
                        elif i < change_point:
                            loss_per_sample = z_max if r else -z_y
                            loss = torch.mean(loss_per_sample)
                        elif self.use_dlr:
                            loss = self.dlr_loss(logits, y).mean()
                        else:
                            loss_per_sample = z_max - z_y
                            loss = torch.mean(loss_per_sample)
                        X_adv, nc = adv_check_and_update(X_pgd.detach(), logits, y, nc, X_adv)
                        loss.backward()

                    if self.use_odi and i < 2:
                        alpha = self.epsilon
                    elif i > change_point:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i-change_point - 1) / (self.num_steps-change_point) * np.pi))
                    else:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i - 1) / (self.num_steps-change_point) * np.pi))
                    eta = alpha * X_pgd.grad.data.sign()
                    X_pgd = X_pgd.detach() + eta.detach()
                    X_pgd = torch.min(torch.max(X_pgd, x - self.epsilon), x + self.epsilon)
                    X_pgd = Variable(torch.clamp(X_pgd, self.v_min, self.v_max), requires_grad=True)
                X_adv, nc = adv_check_and_update(X_pgd, self.model(X_pgd), y, nc, X_adv)

        return X_adv


class MDMTAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_random_starts=1, v_min=0., v_max=1., change_point=25,
                 first_step_size=16./255., seed=0, norm='Linf', num_classes=10,
                 use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_random_starts = num_random_starts
        self.v_min = v_min
        self.v_max = v_max
        self.change_point = change_point
        self.first_step_size = first_step_size
        self.seed = seed
        self.norm = norm
        self.num_classes = num_classes
        self.use_odi = use_odi

    def perturb(self, x_in, y_in):
        change_point = self.change_point
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        nat_logits = self.model(x)
        nat_pred = nat_logits.max(dim=1)[1]
        nat_correct = (nat_pred == y).squeeze()
        nc = torch.zeros_like(y)
        X_adv = x.detach().clone()

        for t in range(9, -1, -1):
            # targets = torch.zeros_like(y)
            # targets += t
            # y_tg = one_hot_tensor(targets, self.num_classes)
            y_gt = one_hot_tensor(y, self.num_classes)
            y_tg = nat_logits.sort(dim=1)[1][:, -t]
            y_tg = one_hot_tensor(y_tg, self.num_classes)

            for _ in range(max(self.num_random_starts, 1)):
                for r in range(2):
                    r_noise = torch.FloatTensor(*x.shape).uniform_(-self.epsilon, self.epsilon).cuda()
                    X_pgd = Variable(x.data+r_noise, requires_grad=True)

                    if self.use_odi:
                        rv = torch.FloatTensor(*nat_logits.shape).uniform_(-1., 1.)
                        rv = rv.to(nat_logits.device)

                    for i in range(self.num_steps):
                        X_pgd.requires_grad_()
                        self.model.zero_grad()
                        with torch.enable_grad():
                            logits = self.model(X_pgd)
                            z_t, _ = torch.max(y_tg * logits - (1 - y_tg) * 1.e8, dim=1)
                            z_y, _ = torch.max(y_gt * logits - (1 - y_gt) * 1.e8, dim=1)
                            if self.use_odi and i < 2:
                                loss = (logits * rv).sum()
                            elif i < 1:
                                loss = torch.mean(z_y)
                            elif i < change_point:
                                loss = torch.mean(z_t) if r else torch.mean(- z_y)
                            else:
                                loss = torch.mean(z_t - z_y)
                            X_adv, nc = adv_check_and_update(X_pgd, logits, y, nc, X_adv)
                        loss.backward()
                        if self.use_odi and i < 2:
                            alpha = self.epsilon
                        elif i > change_point:
                            alpha = self.epsilon * 2.0 * 0.5 * (1 + np.cos((i-change_point-1) / (self.num_steps-change_point) * np.pi))
                        else:
                            alpha = self.epsilon * 2.0 * 0.5 * (1 + np.cos((i-1) / (self.num_steps-change_point) * np.pi))
                        eta = alpha * X_pgd.grad.data.sign()
                        X_pgd = X_pgd.detach() + eta.detach()
                        X_pgd = torch.min(torch.max(X_pgd, x - self.epsilon), x + self.epsilon)
                        X_pgd = Variable(torch.clamp(X_pgd, self.v_min, self.v_max), requires_grad=True)
                    X_adv, nc = adv_check_and_update(X_pgd, self.model(X_pgd), y, nc, X_adv)

        return X_adv
