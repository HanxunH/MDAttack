import os
import util
import time
import argparse
import datasets
import defense
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from autoattack.autoattack import AutoAttack
from attacks.attack_handler import Attacker
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='MD Attacks')
parser.add_argument('--defence', type=str, default='RST')
parser.add_argument('--attack', type=str, default='MD')
parser.add_argument('--n_workers', type=int, default=4)
parser.add_argument('--eps', type=int, default=8)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--datapath', type=str, default='../../datasets')
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--result_path', default='results/')
args = parser.parse_args()
args.eps = args.eps/255
logger = util.setup_logger('MD Attack')


def test(model, testloader):
    model.eval()
    total = 0
    corrects = np.zeros(5)
    with torch.no_grad():
        for data, labels in testloader:
            data = data.to(device)
            outputs = model(data)[-5:]
            predictions = np.array(
                [o.max(1)[1].cpu().numpy() for o in outputs])
            labels = labels.reshape(1, -1).detach().numpy()
            corrects += (predictions == labels).sum(1)
            total += labels.size
    accs = corrects / total
    return accs, total


def main():
    util.build_dirs(args.result_path)
    data = datasets.DatasetGenerator(eval_bs=args.bs, n_workers=args.n_workers,
                                     train_path=args.datapath,
                                     test_path=args.datapath)
    _, test_loader = data.get_loader()
    model = defense.defence_options[args.defence]()
    model = model.to(device)
    model.eval()
    if args.data_parallel:
        print('data_parallel')
        model = torch.nn.DataParallel(model).to(device)
        model.eval()
    for param in model.parameters():
        param.requires_grad = False

    x_test = [x for (x, y) in test_loader]
    x_test = torch.cat(x_test, dim=0)
    y_test = [y for (x, y) in test_loader]
    y_test = torch.cat(y_test, dim=0)
    start = time.time()
    if args.attack == 'AA':
        adversary = AutoAttack(model, norm='Linf', eps=args.eps,
                               logger=logger, verbose=True)
        adversary.set_version(args.attack)
        rs = adversary.run_standard_evaluation(x_test, y_test, bs=args.bs)
        clean_accuracy, robust_accuracy = rs
    else:
        adversary = Attacker(model, epsilon=args.eps, num_classes=10,
                             data_loader=test_loader, logger=logger,
                             version=args.attack)
        rs = adversary.evaluate()
        clean_accuracy, robust_accuracy = rs
    end = time.time()
    cost = end - start
    payload = {
        'clean_acc': clean_accuracy,
        'adv_acc': robust_accuracy,
        'cost': cost
    }
    print(robust_accuracy)
    filename = '%s_%s.json' % (args.defence, args.attack)
    filename = os.path.join(args.result_path, filename)
    util.save_json(payload, filename)
    return


if __name__ == '__main__':
    main()
