# PaddleGan


# Train
```
python -u tools/main.py --config-file configs/cyclegan-cityscapes.yaml
```

continue train from last checkpoint
```
python -u tools/main.py --config-file configs/cyclegan-cityscapes.yaml --resume your_checkpoint_path
```

# Evaluate
```
python tools/main.py --config-file configs/cyclegan-cityscapes.yaml --evaluate-only --load your_weight_path
```


