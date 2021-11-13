# Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness

Code for Paper ["Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness"](https://arxiv.org/abs/2006.13726) by Xingjun Ma, Linxi Jiang, Hanxun Huang, Zejia Weng, James Bailey, Yu-Gang Jiang


---
## Evaluate MD attack
```script
python main.py --defence [Choice from defence models] \
               --attack [MD, MDMT, MDE] \
               --eps 8 --bs 100
```
- `bs` as batch size.
- `eps` as the epsilon.
- Defence models evaluated in the paper are available in the defence folder.
- The following attacks are implemented `['MD', 'MDMT', 'MDE', 'PGD', 'CW', 'PGD-ODI']`, Auto Attacks aviliable at this [link]( https://github.com/fra31/auto-attack)


---
## Part of the code is based on the following repo:
  - MART: https://github.com/YisenWang/MART
  - TREADES: https://github.com/yaodongyu/TRADES
  - RST: https://github.com/yaircarmon/semisup-adv
  - AutoAttack: https://github.com/fra31/auto-attack
  - ODI-PGD https://github.com/ermongroup/ODS
