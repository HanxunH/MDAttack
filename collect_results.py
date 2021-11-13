import os
import json
import datetime
from tabulate import tabulate

result_path = 'results/'


def load_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def load_time_table(attack_targets, defence_targets, header):
    # Load Results
    rs = []
    for defence in defence_targets:
        attack_rs = []
        for attack in attack_targets:
            filename = "%s_%s.json" % (defence, attack)
            filename = os.path.join(result_path, filename)
            if os.path.isfile(filename):
                data = load_json(filename)
                if len(attack_rs) == 0:
                    attack_rs.append(defence)
                attack_rs.append(round(data['cost']/3600, 2))
            else:
                if len(attack_rs) == 0:
                    attack_rs.append(defence)
                    attack_rs.append('None')
                attack_rs.append('None')
        rs.append(attack_rs)

    # Plot Table
    t = datetime.datetime.now()
    print('\n' + '=' * 30 + str(t) + '=' * 30)
    print(tabulate(rs, headers=header, floatfmt=".2f", stralign="left",
                   numalign="left"))
    print('=' * 60 + '=' * len(str(t)) + '\n')


def load_table(attack_targets, defence_targets, header, use_diff=False):
    # Load Results
    rs = []
    for defence in defence_targets:
        attack_rs = []
        for attack in attack_targets:
            filename = "%s_%s.json" % (defence, attack)
            filename = os.path.join(result_path, filename)
            if os.path.isfile(filename):
                data = load_json(filename)
                if len(attack_rs) == 0:
                    attack_rs.append(defence)
                    attack_rs.append(data['clean_acc'])
                if attack == 'AA':
                    attack_rs.append(round(data['adv_acc']*100, 4))
                else:
                    attack_rs.append(data['adv_acc'])
            else:
                if len(attack_rs) == 0:
                    attack_rs.append(defence)
                    attack_rs.append('None')
                attack_rs.append('None')
        if use_diff:
            diff = sorted(attack_rs[2:])[0] - sorted(attack_rs[2:])[1]
            attack_rs.append(diff)
        rs.append(attack_rs)

    rs = sorted(rs, key=lambda k: float(k[-2]))[::-1]
    # Plot Table
    t = datetime.datetime.now()
    print('\n' + '=' * 30 + str(t) + '=' * 30)
    print(tabulate(rs, headers=header, floatfmt=".2f", stralign="left",
                   numalign="left"))
    print('=' * 60 + '=' * len(str(t)) + '\n')


if __name__ == '__main__':
    attack_targets = ['CW', 'PGD', 'ODI', 'DLR', 'MD']
    attack_targets = ['AA', 'MDE']
    defence_targets = ['RST', 'UAT', 'TRADES', 'MART', 'MMA', 'BAT', 'SAT',
                       'ADVInterp', 'FeaScatter', 'Sense', 'JARN_AT', 'Dynamic',
                       'AWP', 'Overfitting', 'ATHE', 'PreTrain', 'RobustWRN']
    header = ['Model', 'Clean'] + attack_targets + ['Diff']
    load_table(attack_targets=attack_targets,
               defence_targets=defence_targets,
               header=header, use_diff=True)
    header = ['Model'] + attack_targets
    load_time_table(attack_targets=attack_targets,
                    defence_targets=defence_targets,
                    header=header)
