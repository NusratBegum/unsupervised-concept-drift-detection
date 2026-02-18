# Unsupervised Concept Drift Detection - Learning Fork

> 🔗 **Original Repository**: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection)
> Original Paper https://link.springer.com/article/10.1007/s41060-024-00620-y and found here s41060-024-00620-y.pdf
> This is my personal fork for learning and running experiments with unsupervised concept drift detection algorithms.

---

## 📚 Table of Contents

1. [What is Concept Drift?](#-what-is-concept-drift)
2. [Project Overview](#-project-overview)
3. [How the Code Works](#-how-the-code-works)
4. [Directory Structure](#-directory-structure)
5. [Installation Guide](#-installation-guide)
6. [Dataset Setup](#-dataset-setup)
7. [Running the Project](#-running-the-project)
8. [Understanding the Detectors](#-understanding-the-detectors)
9. [Understanding the Metrics](#-understanding-the-metrics)
10. [Dataset Fix Documentation](#-dataset-fix-documentation)
11. [Test Results](#-test-results)
12. [Our Modifications Summary](#-our-modifications-summary)

---

## 🎓 What is Concept Drift?

In machine learning, **concept drift** refers to changes in the probability distributions governing a data stream over time. The paper distinguishes two types:

### Real Concept Drift vs Virtual Concept Drift / Covariate Shift / Feature Drift

| Type | Also Known As | Mathematical Definition | What Changes | Requires Labels? |
|------|---------------|------------------------|--------------|------------------|
| **Real Concept Drift** | Concept Drift | P(y\|X) changes | The relationship between features X and target y | Yes (supervised) |
| **Virtual Concept Drift** | Covariate Shift, Feature Drift | P(X) changes | The distribution of features X | No (unsupervised) |

**Examples:**
- **Real drift**: Spam patterns evolve - what makes an email "spam" changes (the relationship between email features and the spam/not-spam label)
- **Virtual drift / Feature drift**: Email writing styles change - features like word frequency shift, but what constitutes spam may remain the same

### What These Detectors Actually Detect

The detectors in this repository are **fully unsupervised** - they observe **only the features X**, never the labels y. This means they detect **Virtual Concept Drift / Covariate Shift / Feature Drift** (changes in P(X)).

> "By virtue of operating on the feature space only, these unsupervised concept drift detectors **cannot detect concept drift in the posterior distribution** (real drift) **unless it is accompanied by a covariate shift** (feature drift)." — from the paper

In practice, this works well because:
1. Changes in P(X) often correlate with changes in P(y|X)
2. Feature drift can still degrade model performance
3. No labeled data is needed, making it practical for real-time streams

### Why Does It Matter?

If left undetected, drift makes machine learning models unreliable. By detecting drift, we can:
- Retrain models when needed
- Alert operators to investigate changes
- Maintain prediction accuracy over time

---

## 🔭 Project Overview

This repository benchmarks **10 unsupervised concept drift detectors** on **real-world data streams**:

| Detector | Full Name | Key Idea |
|----------|-----------|----------|
| **BNDM** | Bayesian Non-parametric Detection Method | Uses Bayesian statistics to detect distribution changes |
| **CSDDM** | Clustered Statistical Test DDM | Clusters data and uses statistical tests |
| **D3** | Discriminative Drift Detector | Trains classifier to distinguish old vs new data |
| **EDFS** | Ensemble Drift with Feature Subspaces | Uses ensemble of detectors on feature subsets |
| **IBDD** | Image-Based Drift Detector | Converts data to images and detects visual changes |
| **NN-DVI** | Nearest Neighbor Density Variation | Measures density changes using nearest neighbors |
| **OCDD** | One-Class Drift Detector | Uses one-class classification |
| **SPLL** | Semi-Parametric Log Likelihood | Measures likelihood ratio changes |
| **UCDD** | Unsupervised Concept Drift Detection | Uses clustering-based approach |
| **UDetect** | Unsupervised Change Detection | Activity recognition approach |

---

## 🔄 How the Code Works

### High-Level Flow

```
main.py
   │
   ▼
runner.py ─────────────────────────────────────────┐
   │                                               │
   ▼                                               │
config.py (defines which datasets & detectors)    │
   │                                               │
   ▼                                               │
For each (dataset, detector) combination:         │
   │                                               │
   ▼                                               │
ModelOptimizer.optimize()                         │
   │                                               │
   ├──► Stream data sample-by-sample              │
   │       │                                       │
   │       ▼                                       │
   │    detector.update(features) ──► Returns True if drift detected
   │       │                                       │
   │       ▼                                       │
   │    If drift: Reset classifiers              │
   │       │                                       │
   │       ▼                                       │
   │    Train classifiers on sample              │
   │                                               │
   ▼                                               │
Calculate metrics (accuracy, LPD, MTR, etc.)      │
   │                                               │
   ▼                                               │
Save results to CSV ◄──────────────────────────────┘
```

### Key Components Explained

#### 1. **Detector Base Class** (`detectors/base.py`)
```python
class UnsupervisedDriftDetector(ABC):
    def update(self, features: dict) -> bool:
        """
        Feed one sample to the detector.
        Returns True if drift is detected, False otherwise.
        """
```

All detectors implement this interface. You feed samples one by one, and the detector signals when it thinks the data distribution has changed.

#### 2. **Model Optimizer** (`optimization/model_optimizer.py`)
This is the experiment runner. For each detector configuration:
1. Streams data sample-by-sample
2. Calls `detector.update()` for each sample
3. If drift detected: resets the "assisted" classifiers
4. Trains classifiers on each sample
5. Records metrics at the end

#### 3. **Classifiers** (`optimization/classifiers.py`)
Maintains 4 classifiers to evaluate detector quality:
- **Base Hoeffding Tree** - Never reset, ignores drift signals
- **Base Naive Bayes** - Never reset, ignores drift signals
- **Assisted Hoeffding Tree** - Reset when detector signals drift
- **Assisted Naive Bayes** - Reset when detector signals drift

If the **assisted** classifiers perform better, the detector is helpful!

---

## 📁 Directory Structure

```
unsupervised-concept-drift-detection/
│
├── main.py                 # Entry point - starts experiments
├── runner.py               # Runs all detector/dataset combinations
├── config.py               # Configuration: which datasets & detectors to test
├── demo.py                 # ⭐ CREATED BY US - Demo testing ALL datasets
├── add_headers.py          # ⭐ CREATED BY US - Adds headers to USP DS CSVs
├── convert_datasets.py     # Original script to convert .arff to .csv
├── eval.py                 # Evaluation and plotting script
├── requirements.txt        # ✏️ MODIFIED BY US - Changed == to >= for Python 3.13
│
├── datasets/               # Dataset loader classes
│   ├── __init__.py         # Exports all dataset classes
│   ├── insects.py          # INSECTS datasets (10 variants)
│   ├── airlines.py         # ✏️ MODIFIED - Use CSV instead of ARFF
│   ├── chess.py            # ✏️ MODIFIED - Use CSV, column names at1-at8
│   ├── electricity.py      # ✏️ MODIFIED - Use CSV instead of ARFF
│   ├── intrusion_detection.py # ✏️ MODIFIED - Use CSV instead of ARFF
│   ├── keystroke.py        # ✏️ MODIFIED - Use CSV instead of ARFF
│   ├── ...                 # Other dataset loaders
│   └── files/              # CSV data files (from USP DS Repository)
│       └── *.csv           # ✏️ Headers added by add_headers.py
│
├── detectors/              # Drift detection algorithms
│   ├── __init__.py         # Exports all detector classes
│   ├── base.py             # Abstract base class
│   ├── d3.py               # D3 - Discriminative Drift Detector
│   ├── ibdd.py             # IBDD - Image-Based Drift Detector
│   ├── spll.py             # SPLL - Semi-Parametric Log Likelihood
│   ├── ...                 # Other detectors
│
├── metrics/                # Performance measurement
│   ├── metrics.py          # Main metrics calculation
│   ├── drift.py            # MTR, MTFA, MTD, MDR calculations
│   └── lift_per_drift.py   # LPD calculation
│
├── optimization/           # Experiment infrastructure
│   ├── model_optimizer.py  # Main experiment runner
│   ├── classifiers.py      # HoeffdingTree & NaiveBayes classifiers
│   ├── config_generator.py # Generates parameter combinations
│   ├── logger.py           # Saves results to CSV
│   └── parameter.py        # Parameter range definitions
│
├── eval/                   # Result analysis
│   ├── cleaner.py          # Cleans result files
│   ├── plotter.py          # Generates plots
│   ├── summarize.py        # Summarizes results
│   └── parser.py           # Parses result files
│
├── results/                # Raw experiment results (CSV files)
│   ├── Elec2/
│   ├── InsectsAbruptBalanced/
│   └── ...
│
└── test/                   # Unit tests
    ├── datasets/           # Dataset tests
    ├── detectors/          # Detector tests
    ├── metrics/            # Metrics tests
    ├── optimization/       # Optimization tests
    └── integration/        # Integration tests
```

---

## 🛠 Installation Guide

### Prerequisites
- Python 3.8+ (tested with 3.13)
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/NusratBegum/unsupervised-concept-drift-detection.git
cd unsupervised-concept-drift-detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate   # On Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Run a quick test (works without datasets)
python -m unittest test.detectors.test_d3
```

Expected output:
```
.
----------------------------------------------------------------------
Ran 1 test in 0.012s

OK
```

---

## 📊 Dataset Setup

### About the USP DS Repository

The datasets come from the **USP Data Stream Repository** maintained by researchers at University of São Paulo. The repository contains real-world data streams with known concept drift points.

**Download Link**: [USP DS Repository](https://sites.google.com/view/uspdsrepository)

### Step-by-Step Dataset Setup (What We Did)

#### 1. Download and Extract
```bash
# Downloaded USP DS Repository.zip from the website
# Extracted the zip file
```

#### 2. Copy CSV Files to `datasets/files/`
We copied all CSV files from the extracted folder directly into `datasets/files/`:

```bash
# Copy all CSV files from the extracted folder
cp "USP DS Repository/Old datasets/"*.csv datasets/files/
cp "USP DS Repository/New datasets/"*.csv datasets/files/
```

Files we copied:
- `INSECTS-abrupt_balanced_norm.csv`
- `INSECTS-gradual_balanced_norm.csv`
- `NOAA.csv`
- `outdoor.csv`
- `elec.csv`
- `poker-lsn.csv`
- `powersupply.csv`
- `rialto.csv`
- `luxembourg.csv`
- `ozone.csv`
- And more...

#### 3. Run the Header Script (Created by Us)
The USP DS Repository CSV files **don't have headers**. We created `add_headers.py` to add proper header rows:

```bash
python add_headers.py
```

This script:
- Reads each CSV file
- Adds the correct header row (column names that match what the loaders expect)
- Saves the file back

#### 4. Dataset Loader Fixes (Done by Us)
Some dataset loaders expected `.arff` format but USP DS provides `.csv`. We modified:
- `datasets/airlines.py` - Use CSV, fix string column types
- `datasets/chess.py` - Use CSV, 8 features (at1-at8)
- `datasets/electricity.py` - Use CSV instead of ARFF
- `datasets/intrusion_detection.py` - Use CSV instead of ARFF
- `datasets/keystroke.py` - Use CSV instead of ARFF

#### 4. Verify Dataset Setup

```bash
# Run all tests - should show 105 tests passing
python -m unittest discover -s test -t .
```

Expected output:
```
....................................s...........................................
.........................
----------------------------------------------------------------------
Ran 105 tests in 11.050s

OK (skipped=1)
```

---

## 🚀 Running the Project

### Option 1: Run the Demo (Recommended for Learning)

```bash
python demo.py
```

This tests D3 detector on **ALL** available datasets:

```
======================================================================
   UNSUPERVISED CONCEPT DRIFT DETECTION - FULL DEMO
   Testing D3 detector on ALL available datasets
======================================================================

──────────────────────────────────────────────────────────────────────
Testing: INSECTS Abrupt Balanced
──────────────────────────────────────────────────────────────────────
  Samples: 52,848 | Features: 33
  Ground truth drifts: [14352, 19500, 33240, 38682, 39510]
  Processed: 20,000 samples
  Drifts detected: 4
  Evaluation: 2/2 ground truth drifts detected

... (tests all 13 datasets)

======================================================================
   SUMMARY
======================================================================

Dataset                                       Status          Drifts
----------------------------------------------------------------------
INSECTS Abrupt Balanced                       ✅ Success       4
INSECTS Gradual Balanced                      ✅ Success       3
INSECTS Incremental Balanced                  ✅ Success       0
INSECTS Incremental-Abrupt Balanced           ✅ Success       1
INSECTS Incremental-Reoccurring Balanced      ✅ Success       0
Electricity                                   ✅ Success       87
NOAA Weather                                  ✅ Success       83
Outdoor Objects                               ✅ Success       11
Poker Hand                                    ✅ Success       96
Powersupply                                   ✅ Success       26
Rialto Bridge Timelapse                       ✅ Success       65
Luxembourg                                    ✅ Success       1
Ozone                                         ✅ Success       12

======================================================================
Demo complete! All datasets tested with D3 detector.
======================================================================
```

### Option 2: Run Tests

```bash
# Run all tests
python -m unittest discover -s test -t .

# Run specific test suites
python -m unittest test.detectors           # All detector tests
python -m unittest test.datasets.test_insects  # INSECTS tests
python -m unittest test.metrics             # Metrics tests
```

### Option 3: Run Full Experiments

```bash
# Run full experiment suite (takes a long time!)
python main.py my_experiment

# With limited threads
OMP_NUM_THREADS=4 python main.py my_experiment
```

Results are saved to `results/<dataset>/<detector>_my_experiment.csv`

### Option 4: Evaluate Results

```bash
python eval.py
```

This generates:
- Summary statistics
- Plots and figures
- Best configuration rankings

---

## 🔬 Understanding the Detectors

### Example: D3 (Discriminative Drift Detector)

**Location**: `detectors/d3.py`

**How it works**:
```
┌─────────────────────────────────────────────────────────────┐
│                    D3 Detection Process                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Collect reference samples (old data window)             │
│     ┌───┬───┬───┬───┬───┬───┬───┬───┐                      │
│     │ ● │ ● │ ● │ ● │ ● │ ● │ ● │ ● │  ← 200 samples       │
│     └───┴───┴───┴───┴───┴───┴───┴───┘                      │
│                                                             │
│  2. Collect recent samples (new data window)                │
│                         ┌───┬───┬───┬───┐                  │
│                         │ ○ │ ○ │ ○ │ ○ │  ← 100 samples   │
│                         └───┴───┴───┴───┘                  │
│                                                             │
│  3. Label them: reference=0, recent=1                       │
│                                                             │
│  4. Train a classifier to distinguish them                  │
│     (Logistic Regression)                                   │
│                                                             │
│  5. Calculate AUC score                                     │
│     - AUC ≈ 0.5 → Can't distinguish → NO DRIFT             │
│     - AUC > 0.7 → Can distinguish → DRIFT DETECTED!        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Intuition**: If a classifier can tell old data from new data, they must be different!

### Detector Parameters

Each detector has tunable parameters. Example for D3:

```python
detector = DiscriminativeDriftDetector2019(
    n_reference_samples=200,       # Size of "old" data window
    recent_samples_proportion=0.5, # Size of "new" window (relative)
    threshold=0.7,                 # AUC threshold for detection
    seed=42                        # Random seed for reproducibility
)
```

---

## 📈 Understanding the Metrics

### Key Metrics

| Metric | Full Name | What It Measures |
|--------|-----------|------------------|
| **acc (ht-dd)** | Accuracy (Hoeffding Tree with Drift Detector) | Classification accuracy with drift detection |
| **acc (ht-no dd)** | Accuracy (Hoeffding Tree, no Drift Detector) | Baseline accuracy without drift detection |
| **lpd (ht)** | Lift Per Drift | Accuracy improvement per detected drift |
| **mtr** | Mean Time to Reaction | Average delay to detect a drift |
| **mtfa** | Mean Time between False Alarms | How often false alarms occur |
| **mtd** | Mean Time to Detection | Time from drift to detection |
| **mdr** | Missed Detection Rate | Proportion of drifts not detected |

### Interpretation

- **Higher is better**: `acc`, `lpd`, `mtfa`
- **Lower is better**: `mtr`, `mtd`, `mdr`

### Example Results

```
Actual drift at: 14,352
Detected at:     14,439
Delay (MTR):     +87 samples

If delay is small → Good detection!
If delay is large or negative → Poor detection
```

---

## 🔧 Dataset Fix Documentation

### The Problem

The original code was written for `.arff` files (ARFF format includes headers). The USP DS Repository provides `.csv` files **without headers**.

### Issues Found

| Issue | Description |
|-------|-------------|
| **Missing headers** | CSV files from USP DS have raw data only, no column names |
| **Different column names** | Code expected specific names like `Att1, Att2, ...` |
| **Format mismatch** | Some loaders expected `.arff`, USP DS has `.csv` |
| **Sample count differences** | USP DS versions have different row counts |

### Fixes Applied

#### 1. Created `add_headers.py`
This script adds proper headers to all CSV files:

```python
# Example of what it does:
# Before: 19.8,14,1019.6,8.4,9.9,15.9,28.9,14,1
# After:  attribute1,attribute2,...,class (header row)
#         19.8,14,1019.6,8.4,9.9,15.9,28.9,14,1
```

#### 2. Updated Dataset Loaders
Modified these files to use CSV format:

| File | Change |
|------|--------|
| `datasets/airlines.py` | Use CSV, fix string column types |
| `datasets/chess.py` | Use CSV, update column names to `at1-at8` |
| `datasets/electricity.py` | Use CSV instead of ARFF |
| `datasets/intrusion_detection.py` | Use CSV instead of ARFF |
| `datasets/keystroke.py` | Use CSV instead of ARFF |

#### 3. Updated `requirements.txt`
Changed from exact versions (`==`) to minimum versions (`>=`) for Python 3.13 compatibility:

```
matplotlib>=3.6.3
numpy>=1.23.1
pandas>=1.4.3
river>=0.11.1
scipy>=1.8.1
scikit-learn>=1.1.1
```

---

## ✅ Test Results

### Current Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| Detectors | 34 | ✅ All pass |
| Metrics | 8 | ✅ All pass |
| Optimization | 34 | ✅ All pass |
| Integration | 3 | ✅ All pass |
| Datasets | 26 | ✅ All pass (1 skipped) |
| **Total** | **105** | ✅ **All pass** |

### Working Datasets

| Dataset | Samples | Features | Has Ground Truth Drifts |
|---------|---------|----------|------------------------|
| INSECTS Abrupt Balanced | 52,848 | 33 | ✅ Yes |
| INSECTS Gradual Balanced | 24,150 | 33 | ✅ Yes |
| INSECTS Incremental Balanced | 57,018 | 33 | ✅ Yes |
| INSECTS Incremental-Abrupt Balanced | 79,986 | 33 | ✅ Yes |
| INSECTS Incremental-Reoccurring Balanced | 79,986 | 33 | ✅ Yes |
| NOAA Weather | 18,159 | 8 | ❌ No |
| Outdoor Objects | 4,000 | 21 | ❌ No |
| Electricity | 45,312 | 8 | ❌ No |
| Poker Hand | 829,201 | 10 | ❌ No |
| Powersupply | 29,928 | 2 | ❌ No |
| Sensor Stream | 2,219,803 | 5 | ❌ No |
| And more... | | | |

---

## 📝 Quick Reference

### Common Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run demo
python demo.py

# Run all tests
python -m unittest discover -s test -t .

# Run specific detector test
python -m unittest test.detectors.test_d3

# Run experiment
python main.py experiment_name

# Evaluate results
python eval.py
```

### File Locations

| What | Where |
|------|-------|
| Dataset CSV files | `datasets/files/*.csv` |
| Experiment results | `results/<dataset>/<detector>.csv` |
| Demo script | `demo.py` |
| Header fixer | `add_headers.py` |

---

## � Our Modifications Summary

This section documents all files **created** or **modified** by us in this learning fork.

### Files Created (New)

| File | Purpose |
|------|---------|
| `demo.py` | Demo script testing D3 detector on ALL 13 datasets |
| `add_headers.py` | Script to add headers to USP DS Repository CSV files |

### Files Modified

| File | What Changed |
|------|-------------|
| `requirements.txt` | Changed `==` to `>=` for Python 3.13 compatibility |
| `datasets/airlines.py` | Use CSV instead of ARFF, fix string column types |
| `datasets/chess.py` | Use CSV, update to 8 features (at1-at8), 503 samples |
| `datasets/electricity.py` | Use CSV instead of ARFF |
| `datasets/intrusion_detection.py` | Use CSV instead of ARFF |
| `datasets/keystroke.py` | Use CSV instead of ARFF |
| `test/datasets/test_chess.py` | Updated expected features/samples |
| `README.md` | Complete rewrite with documentation |
| `.gitignore` | Added .DS_Store, __MACOSX/, USP DS Repository folder |

### Dataset Files (Headers Added)

All CSV files in `datasets/files/` had headers added by `add_headers.py`:
- INSECTS datasets (10 variants): `Att1-Att33,class`
- NOAA.csv: `attribute1-attribute8,class`
- outdoor.csv: `att1-att21,class`
- luxembourg.csv: `att1-att31,class`
- powersupply.csv: `attribute0,attribute1,class`
- ozone.csv: `V1-V72,Class`
- rialto.csv: `att1-att27,class`
- poker-lsn.csv: `s1,r1,s2,r2,...,s5,r5,class`
- elec.csv: `date,day,period,nswprice,nswdemand,vicprice,vicdemand,transfer,class`
- And more...

---

## �🔗 References

- Original Paper: [A benchmark and survey of fully unsupervised concept drift detectors](https://link.springer.com/article/10.1007/s41060-024-00620-y)
- Original Repository: [DFKI-NI/unsupervised-concept-drift-detection](https://github.com/DFKI-NI/unsupervised-concept-drift-detection)
- USP DS Repository: [sites.google.com/view/uspdsrepository](https://sites.google.com/view/uspdsrepository)

---

## 📄 License

BSD 3-Clause License - See [LICENSE](LICENSE) file.
