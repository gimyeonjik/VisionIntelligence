## Overview
This repository contains the supervised-learning challenge submission for 
**DAI3004: Learning Vision Intelligence** at Hanyang University ERICA. 
We train a DHVT-T image classifier 
on CIFAR-100 entirely from scratch, under the course's strict compute budget 
of a single NVIDIA A5000 (24GB) GPU and a wall-clock training time of 
approximately 24 hours per run.

## Structure
```
VisionIntelligence/
├── src/
│   ├── __init__.py
│   ├── dhvt.py
│   ├── data.py
│   ├── losses.py
│   └── schedule.py
├── constants.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Results (Updating...)
| Seed | Top-1 Accuracy | Super-Class Accuracy |
|------|---------------:|---------------------:|
| 0    | 85.19%         | 90.92%               |
| 1    | 00.00%         | 00.00%               |
| 2    | 00.00%         | 00.00%               |
| **Mean ± Std** | **00.00 ± 0.00** | **00.00 ± 0.00** |

## Installation
```
git clone https://github.com/gimyeonjik/VisionIntelligence.git

cd VisionIntelligence

pip install -r requirements.txt
```

## Train
```
# Set Seed you want to set to N
python3 train.py --seed {N} --save_dir ./checkpoints/seed{N}
```

## Evaluation
```
python3 evaluate.py --checkpoint ./checkpoints/seed{N}/stage2_best.pt --data_root ./data
```
