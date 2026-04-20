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
python3 train.py --seed {n} --save_dir ./checkpoints/seed{n}
```

Evaluation
---
```
python3 evaluate.py --checkpoint ./checkpoints/seed{n}/stage2_best.pt --data_root ./data
```
