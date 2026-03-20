# Emotion-Detection
Deep Learning project for AI 535.

## Primary workflow (Notebook)

Use `emotion_detection_experiments.ipynb` as the main experiment runner.

It implements:
- ResNet-18 and ResNet-50 transfer learning
- Frozen backbone and fine-tuning strategies
- Augmentation vs no-augmentation comparison
- Accuracy, precision, recall, and F1 evaluation
- Confusion matrix plots
- Misclassified sample analysis
- Final comparison table across experiments

## Dataset

Expected class folders:

```
dataset_root/
	angry/
	happy/
	relaxed/
	sad/
```

In the notebook config cell:
- set `DATASET_ROOT` to your local dataset path, or
- leave it empty to use the `kagglehub` download fallback.

## Run order

Execute notebook cells top-to-bottom:
1. Setup and imports
2. Constants and paths
3. Data structures
4. Core pipeline functions
5. Minimal run and full experiment matrix
6. Assertions/debug checks

For a quick smoke run, keep `RUN_FULL_EXPERIMENTS = False`.
For full project runs, set `RUN_FULL_EXPERIMENTS = True`.
