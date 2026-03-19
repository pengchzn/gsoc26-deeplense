# DeepLense GSoC Screening Tasks

This repository contains my solutions to two DeepLense GSoC screening tasks:

1. **Common Test I: Multi-Class Classification**
2. **Specific Test V: Lens Finding & Data Pipelines**

Both solutions are implemented in **PyTorch**. I kept the pipelines lightweight, reproducible, and strongly aligned with the official evaluation target: **ROC curves and AUC/AUROC**. Instead of optimizing only for accuracy, model selection is driven by validation AUC, and the final reports include test-time augmentation and ROC-based analysis.

## Repository Contents

- `test1/test1-v3.ipynb`: solution for Common Test I
- `test5/test5-v2.ipynb`: solution for Specific Test V

## Results At A Glance

| Task | Main evaluation setting | Key result |
| --- | --- | --- |
| Test I | Internal validation, TTA | **Macro AUC = 0.9967**, Accuracy = 0.9700 |
| Test I | Holdout split, TTA (supplemental) | **Macro AUC = 0.9961**, Accuracy = 0.9683 |
| Test V | Test set, no TTA | **AUROC = 0.9869**, AP = 0.8117 |
| Test V | Test set, D4 TTA | **AUROC = 0.9886**, AP = 0.8338 |

## Common Design Choices

- **PyTorch over Keras**: chosen for flexible control over custom data pipelines, augmentation, class-imbalance handling, and test-time augmentation.
- **AUC-centric model selection**: checkpoints are selected by validation AUC/AUROC because that is the official metric for both tasks.
- **Compact custom residual CNNs**: I preferred task-specific lightweight ResNet-style models over heavier pretrained backbones, which keeps the models efficient while still performing strongly on astronomy images.
- **Training-only normalization**: normalization statistics are computed from the training split only to avoid leakage.

## Test I: Multi-Class Classification

### Problem Setup

The task is to classify single-channel galaxy images into three classes:

- `no`: strong lensing image with no substructure
- `sphere`: subhalo substructure
- `vort`: vortex substructure

The dataset is balanced. In my pipeline, I used the provided `dataset/train` split for learning, created a **stratified 90/10 internal validation split** for model selection, and kept the provided `dataset/val` split as an unseen **holdout set** for supplemental reporting. The resulting sizes were:

- train: **27,000**
- internal validation: **3,000**
- holdout: **7,500**

### Method

- **Input**: grayscale `.npy` images of shape `1 x 150 x 150`
- **Normalization**: scalar mean/std estimated from 5,000 random training samples
- **Training augmentation**:
  - random horizontal and vertical flips
  - random rotations
  - small affine translations
  - Gaussian blur
- **Model**: custom ResNet-style CNN with residual blocks and a 2-layer classification head
  - parameter count: **2,828,259**
- **Loss**: cross-entropy with **label smoothing** (`epsilon = 0.05`)
- **Optimizer / schedule**:
  - `AdamW`
  - `OneCycleLR`
  - gradient clipping
  - early stopping
- **Inference enhancement**: **8-view test-time augmentation (TTA)**

### Workflow

1. Load the three class folders and build train / internal-validation / holdout splits.
2. Compute training-only normalization statistics.
3. Train a compact residual CNN with augmentation and AUC-based checkpoint selection.
4. Evaluate first on the internal validation split, then on the holdout split.
5. Report per-class ROC curves, macro AUC, confusion matrices, and classification reports.

### Results

**Internal validation (standard inference)**

- Accuracy: **0.9570**
- Macro AUC: **0.9938**

**Internal validation (8-view TTA)**

- Accuracy: **0.9700**
- Macro AUC: **0.9967**
- Per-class AUC:
  - `no`: **0.9968**
  - `sphere`: **0.9947**
  - `vort`: **0.9986**

**Holdout split (8-view TTA, supplemental)**

- Accuracy: **0.9683**
- Macro AUC: **0.9961**
- Per-class AUC:
  - `no`: **0.9961**
  - `sphere`: **0.9936**
  - `vort`: **0.9986**

### Takeaway

For Test I, a lightweight residual CNN plus moderate augmentation and TTA was sufficient to achieve very strong multi-class separability. The main improvement came from aligning checkpoint selection and final evaluation with the task metric, namely macro ROC-AUC.

## Test V: Lens Finding & Data Pipelines

### Problem Setup

The task is to identify whether an object is a lens or non-lens from **3-channel observational images** with shape `(3, 64, 64)`. The central difficulty here is **strong class imbalance**:

- training positive fraction: **5.69%**
- test positive fraction: **0.99%**

Because of this imbalance, I treated **AUROC as the primary metric** and also reported **Average Precision (AP)** as a complementary metric.

Split sizes:

- train: **27,364**
- validation: **3,041**
- test: **19,650**

### Method

- **Input**: multi-filter `.npy` images of shape `3 x 64 x 64`
- **Data split**:
  - train on `train_lenses` + `train_nonlenses`
  - build a **stratified 90/10 validation split**
  - keep `test_lenses` + `test_nonlenses` untouched for final evaluation
- **Normalization**: channel-wise mean/std computed from the training split only
- **Training augmentation**:
  - random horizontal and vertical flips
  - random 90-degree rotations
- **Model**: custom ResNet-style binary classifier
  - parameter count: **2,795,297**
  - retained over heavier pretrained alternatives because the lightweight residual model performed more reliably on this dataset
- **Imbalance handling**:
  - weighted BCE with `pos_weight = 16.57`
  - label smoothing (`epsilon = 0.05`)
  - Mixup (`alpha = 0.2`)
  - focal loss also implemented as an ablation option, but the final model uses weighted BCE
- **Optimization**:
  - `AdamW`
  - `ReduceLROnPlateau`
  - AMP mixed-precision training
  - early stopping
- **Inference enhancement**: **D4 test-time augmentation** using 8 geometric symmetries

### Workflow

1. Load the lens and non-lens directories and construct a stratified validation split.
2. Compute channel-wise normalization on the training split.
3. Train a binary residual CNN with weighted loss and Mixup to address imbalance.
4. Select the best checkpoint using validation AUROC.
5. Evaluate on the official test set with and without D4 TTA.
6. Report ROC and PR curves, AUROC, AP, and a threshold analysis for interpretation.

### Results

| Split | Setting | AUROC | AP |
| --- | --- | ---: | ---: |
| Validation | No TTA | 0.9958 | 0.9551 |
| Test | No TTA | 0.9869 | 0.8117 |
| Validation | D4 TTA | 0.9968 | 0.9619 |
| Test | D4 TTA | **0.9886** | **0.8338** |

Additional threshold analysis on the **test D4-TTA** predictions:

- optimal threshold by Youden's J: **0.7216**
- recall: **0.9590**
- precision: **0.2275**

This threshold behavior is expected for a highly imbalanced problem: the model captures most positives, while precision remains constrained by the very large number of non-lenses.

### Takeaway

For Test V, the most important design choice was not a larger backbone, but a pipeline that explicitly handles imbalance. Weighted BCE, Mixup, AUROC-based model selection, and D4 TTA together produced a strong final test result of **AUROC 0.9886**.

## Final Remarks

These two tasks were approached with the same principle: build a reliable, metric-aligned pipeline first, then add only the task-specific components that are clearly justified by the data.

- For **Test I**, that meant a clean multi-class classifier with macro-AUC-driven selection.
- For **Test V**, that meant explicit imbalance handling and stronger inference-time averaging.

The notebooks are organized as end-to-end experiments and include data loading, preprocessing, training, evaluation, ROC analysis, and saved artifacts for reproducibility.
