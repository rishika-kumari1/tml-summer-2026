# Stolen Model Detection – Assignment 2

## Overview

This repository contains our solution for the **Stolen Model Detection** task from the Trustworthy Machine Learning 2026 course.

The objective is to assign a stealing confidence score to each of the 360 suspect models, indicating how likely it is that the model was copied or derived from the provided target model.

---

## Setup

### 1. Connect to the HPC Cluster

```bash
ssh <username>@conduit.hpc.uni-saarland.de
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv task2_env
source task2_env/bin/activate
```

### 3. Install Required Packages

```bash
pip install torch torchvision pandas requests safetensors
```

### 4. Download Assignment Files

Clone the repository or download the task files from the provided HuggingFace repository.

The directory structure should be:

```text
tml26_task2/
├── detect.py
├── submission.py
├── target_model/
│   ├── weights.safetensors
│   └── train_main_idx.json
├── suspect_models/
│   ├── suspect_000.safetensors
│   ├── suspect_001.safetensors
│   └── ...
```

---

## Running the Code

Activate the virtual environment and run:

```bash
source task2_env/bin/activate

cd tml26_task2

python detect.py
```

The script computes similarity scores for all suspect models and generates a submission file.

---

## Method

The detector combines both structural and behavioral similarity signals.

For each suspect model, the following metrics are computed:

- Weight Similarity
- Batch Normalization Similarity
- Prediction Agreement
- KL-Divergence Similarity
- Feature Representation Similarity

The final stealing confidence score is calculated as:

```text
Score =
0.10 × Weight Similarity +
0.10 × BN Similarity +
0.30 × Prediction Agreement +
0.25 × KL Similarity +
0.25 × Feature Similarity
```

The resulting scores are normalized using z-score normalization followed by a sigmoid transformation.

---

## Output

Running `detect.py` generates:

```text
submission.csv
```

Submission format:

```text
id,score
0,...
1,...
...
359,...
```

The file contains one confidence score for every suspect model.

---

## Leaderboard Submission

Edit the following values in `submission.py`:

```python
API_KEY = "YOUR_API_KEY"
FILE_PATH = "submission.csv"
```

Submit the results using:

```bash
python submission.py
```

---

## Notes

- Uses GPU if available
- Uses the CIFAR-100 normalization values provided in the assignment
- Evaluates 1000 training samples and 1000 test samples
- Generates exactly 360 confidence scores
- Produces a valid `submission.csv` file for leaderboard submission
- Requires the target model weights, suspect model weights, and `train_main_idx.json`
