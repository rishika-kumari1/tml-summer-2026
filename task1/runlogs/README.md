# Membership Inference Attack – Assignment 1

## Overview

This repository contains the implementation of a membership inference attack for the Trustworthy Machine Learning 2026 course.

The goal is to assign a confidence score to each sample indicating whether it was part of the model’s training dataset.

---

## Setup

### 1. Connect to HPC cluster

```bash
ssh <username>@conduit.hpc.uni-saarland.de
```

### 2. Create working directory

```bash
mkdir ~/tml26_task1
cd ~/tml26_task1
```

### 3. Download assignment files

```bash
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/task_template.py"
```

---

## Running the Code

### Interactive testing

```bash
condor_submit -i mia.sub
```

Then:

```bash
cd /home/<username>/tml26_task1
python task_template.py
```

---

## Submit Job

```bash
mkdir -p runlogs
condor_submit mia.sub
```

---

## Method

We use an augmentation-based approach to compute membership scores.

For each sample:

* Generate multiple augmented versions (random flip + rotation)
* Compute cross-entropy loss for each version
* Average the loss across augmentations
* Use the negative average loss as the membership score

Final score:

```python
score = - mean(loss over augmentations)
```

Scores are passed through a sigmoid function to ensure values are in the range [0,1].

---

## Output

The script generates:

```bash
submission.csv
```

Format:

```csv
id,score
```

---

## Notes

* Uses GPU if available
* Requires correct normalization (provided in code)
* Uses absolute paths to avoid cluster issues
* API key must be added in `task_template.py`
