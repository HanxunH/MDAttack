import torch
import numpy as np
from . import PGD, MD, autopgd_pt, fab_pt
from .utils import adv_check_and_update

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Attacker():
    def __init__(self, model, epsilon=8./255., v_min=0., v_max=1.,
                 num_classes=10, data_loader=None, logger=None,
                 version='MD', verbose=True):
        self.model = model
        self.epsilon = epsilon
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.data_loader = data_loader
        self.logger = logger
        self.verbose = verbose
        self.md = MD.MDAttack(model, epsilon, num_steps=50, step_size=2/255,
                              v_min=v_min, v_max=v_max, change_point=20,
                              first_step_size=16./255., seed=0, norm='Linf',
                              num_classes=num_classes, use_odi=False)
        self.md_dlr = MD.MDAttack(model, epsilon, num_steps=50, step_size=2/255,
                                  v_min=v_min, v_max=v_max, change_point=20,
                                  first_step_size=16./255., seed=0, norm='Linf',
                                  num_classes=num_classes, use_odi=False,
                                  use_dlr=True)
        self.mdmt = MD.MDMTAttack(model, epsilon, num_steps=50, step_size=2/255,
                                  v_min=v_min, v_max=v_max, change_point=20,
                                  first_step_size=16./255., seed=0, norm='Linf',
                                  num_classes=num_classes, use_odi=False)
        self.mdmt_plus = MD.MDMTAttack(model, epsilon, num_steps=50,
                                       num_random_starts=10, step_size=2/255,
                                       v_min=v_min, v_max=v_max, change_point=20,
                                       first_step_size=16./255., seed=0,
                                       norm='Linf', num_classes=num_classes)
        self.pgd = PGD.PGDAttack(model, epsilon, num_steps=20, step_size=0.8/255.,
                                 num_restarts=5, v_min=v_min, v_max=v_max,
                                 num_classes=num_classes, random_start=True,
                                 type='PGD', use_odi=False)
        self.mt = PGD.MTPGDAttack(model, epsilon, num_steps=20, step_size=0.8/255,
                                  num_restarts=5, v_min=v_min, v_max=v_max,
                                  num_classes=num_classes, random_start=True,
                                  use_odi=False)
        self.odi = PGD.PGDAttack(model, epsilon, num_steps=20, step_size=0.8/255.,
                                 num_restarts=5, v_min=v_min, v_max=v_max,
                                 num_classes=num_classes, random_start=True,
                                 type='PGD', use_odi=True)
        self.cw = PGD.PGDAttack(model, epsilon, num_steps=100, step_size=0.8/255,
                                num_restarts=1, v_min=v_min, v_max=v_max,
                                num_classes=num_classes, random_start=True,
                                type='CW', use_odi=False)
        self.apgd = autopgd_pt.APGDAttack(model, eps=epsilon, device=device)
        self.apgd_mt = autopgd_pt.APGDAttack_targeted(model, eps=epsilon,
                                                      device=device)
        self.fab = fab_pt.FABAttack_PT(model, n_restarts=5, n_iter=100,
                                       eps=epsilon, seed=0,
                                       verbose=False, device=device)
        self.fab.targeted = True
        self.fab.n_restarts = 1

        self.attacks_to_run = []

        if version == 'MD':
            self.attacks_to_run = [self.md]
        elif 'MD_first_stage_step_search' in version:
            if '_05' in version:
                self.md.change_point = 5
            else:
                change_point = int(version[-2:])
                self.md.change_point = change_point
            self.attacks_to_run = [self.md]
        elif 'MD_first_stage_initial_step_size_search' in version:
            eps = int(version[-2:])
            self.md.initial_step_size = eps/255
            self.attacks_to_run = [self.md]
        elif version == 'MD_DLR':
            self.attacks_to_run = [self.md_dlr]
        elif version == 'MDMT':
            self.attacks_to_run = [self.mdmt]
        elif version == 'PGD':
            self.attacks_to_run = [self.pgd]
        elif version == 'MT':
            self.attacks_to_run = [self.mt]
        elif version == 'DLR':
            self.apgd.loss = 'dlr'
            self.attacks_to_run = [self.apgd]
        elif version == 'DLRMT':
            self.apgd_mt.loss = 'dlr'
            self.attacks_to_run = [self.apgd_mt]
        elif version == 'ODI':
            self.attacks_to_run = [self.odi]
        elif version == 'CW':
            self.attacks_to_run = [self.cw]
        elif version == 'MDE':
            self.apgd.loss = 'ce'
            self.attacks_to_run = [self.apgd, self.md, self.mdmt, self.fab]
        elif version == 'MDMT+':
            self.attacks_to_run = [self.mdmt_plus]
        else:
            raise('Unknown')

    def evaluate(self):
        clean_count = 0
        adv_count = 0
        total = 0

        for images, labels in self.data_loader:
            images, labels = images.to(device), labels.to(device)
            nc = torch.zeros_like(labels)
            total += labels.shape[0]

            # Check Clean Acc
            with torch.no_grad():
                clean_logits = self.model(images)
                if isinstance(clean_logits, list):
                    clean_logits = clean_logits[-1]
            clean_pred = clean_logits.data.max(1)[1].detach()
            clean_correct = (clean_pred == labels).sum().item()
            clean_count += clean_correct

            # Build x_adv
            x_adv = images.clone()
            x_adv_targets = images.clone()
            x_adv, nc = adv_check_and_update(x_adv_targets, clean_logits,
                                             labels, nc, x_adv)

            # All attacks and update x_adv
            for a in self.attacks_to_run:
                x_p = a.perturb(images, labels).detach()
                with torch.no_grad():
                    adv_logits = self.model(x_p)
                x_adv, nc = adv_check_and_update(x_p, adv_logits, labels,
                                                 nc, x_adv)
            # Robust Acc
            with torch.no_grad():
                adv_logits = self.model(x_adv)
                if isinstance(adv_logits, list):
                    adv_logits = adv_logits[-1]

            adv_pred = adv_logits.data.max(1)[1].detach()
            adv_correct = (adv_pred == labels).sum().item()
            adv_count += adv_correct

            # Log
            if self.verbose:
                rs = (clean_count, total, clean_count * 100 / total,
                      adv_count, total, adv_count * 100 / total)
                payload = (('Clean: %d/%d Clean Acc: %.2f Adv: %d/%d '
                           + 'Adv_Acc: %.2f') % rs)
                self.logger.info('\033[33m'+payload+'\033[0m')

        clean_acc = clean_count * 100 / total
        adv_acc = adv_count * 100 / total
        return clean_acc, adv_acc
