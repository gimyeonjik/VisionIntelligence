Installation
---
```
git clone https://github.com/gimyeonjik/VisionIntelligence.git

cd VisionIntelligence

pip install -r requirements.txt
```

Train
---
```
# Set Seed you want to set to n
python3 train.py --seed {n} --save_dir ./checkpoints/seed{n}
```

Evaluation
---
```
python3 evaluate.py --checkpoint ./checkpoints/seed{n}/stage2_best.pt --data_root ./data
```
