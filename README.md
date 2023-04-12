# Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness

Code for Paper ["Imbalanced Gradients: A Subtle Cause of Overestimated Adversarial Robustness"](https://link.springer.com/article/10.1007/s10994-023-06328-7) by Xingjun Ma, Linxi Jiang, Hanxun Huang, Zejia Weng, James Bailey, Yu-Gang Jiang


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


## Citation
If you use this code in your work, please cite the accompanying paper:
```
@article{ma2023imbalanced,
  title={Imbalanced gradients: a subtle cause of overestimated adversarial robustness},
  author={Ma, Xingjun and Jiang, Linxi and Huang, Hanxun and Weng, Zejia and Bailey, James and Jiang, Yu-Gang},
  journal={Machine Learning},
  pages={1--26},
  year={2023},
  publisher={Springer}
}
```

---
## Part of the code is based on the following repo:
  - MART: https://github.com/YisenWang/MART
  - TREADES: https://github.com/yaodongyu/TRADES
  - RST: https://github.com/yaircarmon/semisup-adv
  - AutoAttack: https://github.com/fra31/auto-attack
  - ODI-PGD https://github.com/ermongroup/ODS
  - Linxi Jiang implementation on MD attack https://github.com/Jack-lx-jiang/MD_attacks
