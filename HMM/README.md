# Human Activity Recognition using Hidden Markov Models
### Group 21 | Accelerometer & Gyroscope | Walking · Standing · Still · Jumping

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Team & Contributions](#team--contributions)
4. [Setup & Installation](#setup--installation)
5. [Data Organisation](#data-organisation)
6. [Running the Notebook](#running-the-notebook)
7. [What the Notebook Produces](#what-the-notebook-produces)
8. [Task Allocation](#task-allocation)

---

## Project Overview

This project implements a **Hidden Markov Model (HMM)** system **entirely from scratch**

| Detail | Value |
|--------|-------|
| Activities | Walking, Standing, Still, Jumping |
| Sensors | Accelerometer (x,y,z) + Gyroscope (x,y,z) |
| Users | Rob, Antony |
| Recordings | 50 labelled zip files (25 per member) |
| Window size | 1 second, non-overlapping |
| Features | 72 per window (36 per sensor) |
| Models | 8 HMMs — one per (activity × sensor) |
| Training | Baum-Welch (log-space EM, \|ΔLL\| < 1e-3) |
| Decoding | Viterbi (log-space dynamic programming) |
| Emissions | sklearn GaussianMixture (2 components, diagonal covariance) |

**Use case:** Outpatient physiotherapy compliance monitoring — a smartphone at waist level automatically logs activity durations so therapists can verify prescribed movement targets remotely.

---

## Repository Structure

```
HiddenMarkovModel-Grp21/HMM
│
├── README.md
├── HMM_Activity_Recognition_Grp21_FINAL.ipynb   ← Main notebook
├── HMM_Report_Group21_FINAL.pdf                 ← Project report
├── Team Task Sheet_Machine Learning Techniques II_Formative 2 - Hidden Markov Models_C1 _Team21.pdf                 ← Task allocation sheet
│
├── Dataset/
│   ├── Rob/                        ← 25 zip files from Rob
│   │   ├── Rob_Walking_01.zip
│   │   ├── Rob_Walking_02.zip
│   │   ├── Rob_Standing_01.zip
│   │   ├── Rob_Still_01.zip
│   │   ├── Rob_Jumping_01.zip
│   │   └── ...
│   │
│   ├── Antony/                     ← 25 zip files from Antony
│   │   ├── Antony_Walking_01.zip
│   │   ├── Antony_Standing_01.zip
│   │   ├── Antony_Still_01.zip
│   │   ├── Antony_Jumping_01.zip
│   │   └── ...
│   │
│   └── NewTestSamples/             ← Unseen records/samples for testing the model performance
│       └── *.zip
│
└── outputs/                        ← Generated after running notebook, mounted to save directly on drive.
    ├── trained_hmm_models.pkl      
    ├── fitted_scaler.pkl           
    ├── features_for_hmm.csv        
    └── fig1–fig9 *.png             
```

### Inside each `.zip` recording

```
Rob_Walking_01.zip
├── Accelerometer.csv    ← columns: time, x, y, z
├── Gyroscope.csv        ← columns: time, x, y, z
└── Metadata.csv         ← contains: device name, sampleRateMs
```

> **Naming convention:** `{Member}_{Activity}_{NN}.zip`  
> Initially, this was a challenge, and we handled it by resolving the activity label from: (1) filename, (2) internal Annotation/Tags CSV, (3) internal entry paths — whichever succeeds first.

---

## Team & Contributions

| Member | Phone Model | Sampling Rate | GitHub |
|--------|-------------|--------------|--------|
| Rob    | Iphone 15 Pro | 100 Hz | @Gatwaza |
| Antony | Samsung | 100 Hz | @tonywahome |

> Both phones target 100 Hz via Sensor Logger settings. The notebook handles different rates automatically — see Section 4b.

---

## Setup & Installation

### Note — Google Colab (Recommended, zero install)

**Step 1 — Upload data to Google Drive**

Create this exact folder structure on your Drive:

```
MyDrive/
└── HiddenMarkovModel Data Grp 21/      ← must match exactly
    ├── Rob/
    ├── Antony/
    └── NewTestSamples/                 ← must match too
```

**Step 2 — Open the notebook in Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-ORG/HiddenMarkovModel-Grp21/blob/main/HMM_Activity_Recognition_Grp21_FINAL.ipynb)

Or: `File → Open notebook → GitHub` → https://github.com/Gatwaza/HiddenMarkovModel-Data-Grp-21.git.

**Step 3 — Run all cells**

```
Runtime → Run all
```

All dependencies (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `sklearn`) are pre-installed in Colab. No `pip install` required.

---

### Option B — Local Jupyter

**Requirements:** Python 3.9+

```bash
# 1. Clone
git clone https://github.com/YOUR-ORG/HiddenMarkovModel-Grp21.git
cd HiddenMarkovModel-Grp21

# 2. Virtual environment
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch notebook
jupyter notebook HMM_Activity_Recognition_Grp21_FINAL.ipynb
```

**`requirements.txt`**
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
scikit-learn>=1.3
jupyter>=1.0
```

> **Local path change required:** Update the `BASE` variable in **Section 2** of the notebook:
> ```python
> BASE = '/path/to/your/local/HiddenMarkovModel Data Grp 21'
> ```
> Also comment out the `drive.mount(...)` line.

---

## Data Organisation

### File count requirements

| Member | Files | Activities covered |
|--------|-------|--------------------|
| Rob | 25 | Walking, Standing, Still, Jumping |
| Antony | 25 | Walking, Standing, Still, Jumping |
| **Total** | **50** | All 4 |

Each activity accumulated **≥ 90 seconds total** across all recordings. Each recording was **5–10 seconds** long. Section 4a of the notebook validates these requirements automatically.

### Naming examples

```
Rob_Walking_01.zip       Rob_Walking_02.zip       ...
Rob_Standing_01.zip      Rob_Standing_02.zip      ...
Rob_Still_01.zip         Rob_Still_02.zip         ...
Rob_Jumping_01.zip       Rob_Jumping_02.zip       ...

Antony_Walking_01.zip    Antony_Walking_02.zip    ...
Antony_Standing_01.zip   ...
```

---

## Running the Notebook

The notebook has **38 cells** across **14 sections**. Run top-to-bottom:

| Section | Cell(s) | What It Does | Key Output |
|---------|---------|-------------|------------|
| 1 — Setup | 2 | Imports, SEED=42, global constants | Constants defined |
| 2 — Drive | 4 | Mount Drive, define all paths | Drive connected |
| 3 — Load | 6 | `process_recording()` — parses all zips, extracts sensor data | `all_recordings_data` list |
| 4 — Validate | 8–10 | 50-file count, duration check, sampling rate table, windowing justification | Validation report printed |
| 5 — Features | 12–13 | Windowing, 72-feature extraction, train/test split, Z-score normalisation | `train_norm`, `test_norm`, scaler saved |
| 6 — Plots | 15–17 | Raw signals (Fig 1), feature boxplots (Fig 2), duration histogram (Fig 3) | 3 PNG files |
| 7 — Save CSV | 19 | Save normalised training features | `features_for_hmm.csv` on Drive |
| 8 — HMM class | 21 | Full `HMM` class: `train`, `viterbi_decode`, `score`, `generate_sequence` | Class defined |
| 9 — Train | 23–24 | Train 8 HMMs (4 activities × 2 sensors), convergence plots (Fig 4) | `hmm_models` dict, Fig 4 |
| 10 — Evaluate | 26–28 | Test set metrics, confusion matrices (Fig 5), Viterbi decoded sequences (Fig 6) | Tables + Fig 5–6 |
| 11 — HMM viz | 30–31 | Transition matrix heatmaps (Fig 7), GMM emission means (Fig 8) | Fig 7–8 |
| 12 — Save | 33 | Save model bundle to Drive | `trained_hmm_models.pkl` |
| 13 — New data | 35–37 | Load/synthesise new samples, evaluate, confusion matrices (Fig 9) | Fig 9 + metrics |
| 14 — Summary | 39 | Performance comparison table, rubric metrics table | Final printed tables |

**Expected runtime:** 10–25 minutes on Colab CPU, depending on dataset size.

---

## What the Notebook Produces

### Saved to Google Drive (`MyDrive/HiddenMarkovModel Data Grp 21/`)

| File | Description |
|------|-------------|
| `trained_hmm_models.pkl` | Bundle: all 8 HMM objects + feat_cols + id_cols + metadata |
| `fitted_scaler.pkl` | StandardScaler fitted on training set only |
| `features_for_hmm.csv` | Z-score normalised feature matrix (training windows, no IDs) |

### Saved to `/content/` (I suggest you download before Colab session ends)

| File | Description |
|------|-------------|
| `fig1_raw_sensor_windows.png` | Raw x/y/z sensor data — first 1-s window per activity per sensor |
| `fig2_feature_boxplots.png` | Normalised feature distributions by activity (8 selected features) |
| `fig3_duration_histogram.png` | Recording duration distribution with 5–10 s target band |
| `fig4_convergence.png` | Baum-Welch log-likelihood convergence for all 8 HMMs |
| `fig5_confusion_test.png` | Confusion matrices — held-out test set (accel + gyro) |
| `fig6_decoded_sequences.png` | Viterbi decoded state paths with raw magnitude overlay |
| `fig7_transition_matrices.png` | Learned A matrices as heatmaps (self-loops highlighted in blue) |
| `fig8_gmm_emission_means.png` | GMM weighted emission means per hidden state (first 12 features) |
| `fig9_confusion_new.png` | Confusion matrices — new unseen samples (accel + gyro) |

---

## Task Allocation

| Task | Rob | Antony |
|------|:---:|:------:|
| Data collection — Walking (≥12 files) | |  |
| Data collection — Standing (≥12 files) |  |  |
| Data collection — Still (≥12 files) |  |  |
| Data collection — Jumping (≥12 files) |  |  |
| `process_recording()` — zip parsing, label resolution, timestamp conversion |  | |
| Section 4 — data validation, sampling rate analysis, windowing justification | |  |
| `window_data()`, `calculate_rms()`, `calculate_sma()` |  | |
| `calculate_axis_correlation()`, `calculate_dominant_frequency()`, `calculate_spectral_energy()` | |  |
| `extract_features_from_window()` — full 72-feature pipeline |  | |
| Train/test split (recording-level), StandardScaler normalisation | |  |
| `HMM.__init__()`, `_initialize_gmm_emissions()`, `_get_log_emission_probs()` |  | |
| `_forward_pass()`, `_backward_pass()`, `_compute_log_gamma_and_xi()` |  | |
| `_update_parameters()` (M-step), `_compute_total_log_likelihood()`, `_clean()` |  | |
| `train()` (Baum-Welch with convergence check), `viterbi_decode()`, `score()`, `generate_sequence()` |  | |
| `build_sequences()`, training loop, convergence plots (Fig 4) | |  |
| Evaluation functions: `predict_activity_hmm()`, `run_evaluation()`, `print_metrics()` |  | |
| Confusion matrices (Fig 5), Viterbi decoded sequence plots (Fig 6) | |  |
| Transition matrix heatmaps (Fig 7), GMM emission bar plots (Fig 8) | |  |
| New unseen samples pipeline, Fig 9, sensor fusion |  | |
| Figures 1–3 (raw signals, boxplots, histograms) | |  |
| Report writing |  |  |
| GitHub repository setup & README |  | |

---

## Reproducibility

```python
SEED = 42
np.random.seed(SEED)
```

All random operations (train/test split, KMeans init, GMM init, synthetic noise generation) use SEED=42. Re-running the notebook top-to-bottom on the same data produces identical results.

---

*Group 21 · HMM Activity Recognition · Sensor Logger App · SEED=42*
