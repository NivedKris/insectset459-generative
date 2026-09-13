# When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale

Official research repository and reproducibility kit for the manuscript:

> **When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale**  
> **Authors:** Nived Krishna$^1$ and Kala S$^{2,*}$, *Senior Member, IEEE*  
> $^1$*Department of Computer Science & Engineering, Indian Institute of Technology Kharagpur, India*  
> $^2$*Department of Electronics & Communication Engineering, Indian Institute of Information Technology Kottayam, India*  
> **Paper & Code:** [https://github.com/NivedKris/insectset459-generative](https://github.com/NivedKris/insectset459-generative)

---

## Abstract

Generative augmentation is widely employed to address extreme class imbalance in long-tail recognition, yet its behavior at extreme taxonomical scale (hundreds of fine-grained species) has remained largely unquantified. In this paper, we evaluate conditional generative models (Optimal Transport Conditional Flow Matching, Diffusion Probabilistic Models, Conditional Variational Autoencoders, and Auxiliary Classifier GANs) on **InsectSet459**, an extreme long-tail benchmark comprising 459 insect species. 

While generators achieve low batch-level distributional distance (Fréchet Audio Distance and Maximum Mean Discrepancy, $\text{MMD}^2 = 0.058 \pm 0.019$), per-sample class-identity verification reveals a severe pathology: **Species-Identity Preservation Rate (SIPR)** collapses from **64.5%** on head species to **18.6%** on tail species (compared to a real-data oracle baseline of **48.6%**). We term this phenomenon **identity collapse** and demonstrate that it is architecture-general across all four generative families. Post-hoc confidence filtering discards 63.1% of generated data, abandoning 35 of 115 tail species entirely. In a pre-registered $N=20$ independent-seed equivalence audit, Two One-Sided Tests (TOST, $\pm 0.020$ margin) prove inconclusive, while paired significance testing ($p = 0.0003, d = 0.98$) demonstrates that even filtered generative augmentation is statistically outperformed by zero-cost spectral masking (SpecAugment).

---

## Key Experimental Findings

1. **Distributional Fidelity Does Not Predict Sample Utility:**
   Batch-level metrics ($\text{MMD}^2$, FAD) fail to detect sample-level class corruption. Across 459 species, Spearman rank correlation between distributional fidelity and identity preservation is null ($\rho = -0.026, p = 0.585$).
2. **Identity Collapse is Architecture-General:**
   Under extensive hyperparameter and learning-rate sweeps, all baseline generative families collapse near 0% top-1 accuracy on tail classes (ACGAN: 0.0%, DDPM: 0.0%, CVAE: 6.7%), while OT-CFM reaches 18.6% (compared to the 48.6% real-data oracle bar).
3. **Genus-Level Acoustic Prior Limitation:**
   Within-genus acoustic cosine distance in PANNs embedding space is only 4.55% smaller than cross-genus distance ($0.174 \text{ vs. } 0.182, p = 0.006, n = 74$ multi-species genera). Synthetic genus confusion (12.2%) exceeds real errors (8.1%, cluster-deflated $z = 2.89, p = 0.004$), but this 4.1 pp gap cannot account for the 30.0 pp collapse.
4. **Oracle Confidence Triage Creates Tail Abandonment:**
   Filtering synthetic samples by oracle confidence retains only 36.9% of samples. In the critical tail tier, 35/115 species receive zero usable synthetic samples, and 59 more receive fewer than 5, reducing effective tail augmentation $K_{\text{gen}}$ from 50 to 2.83.
5. **Rigorous Equivalence and Significance Protocol ($N=20$ Seeds):**
   A pre-registered TOST audit ($\pm 0.020$ margin) yields a 90% CI of $[-0.0202, -0.0088]$ ($p = 0.055$, inconclusive). A paired $t$-test ($p = 0.0003, d = 0.98$) and binomial sign test (16/20 wins, $p = 0.012$) establish that standard zero-cost SpecAugment significantly outperforms filtered generative augmentation.

---

## 🚀 Execution Guide & Operational Modes

This repository can be executed in multiple ways depending on whether the user seeks rapid numerical verification, full retraining from scratch, targeted single-step inspection, or direct Python module execution.

---

### Method 1: Master Universal Runner (`run.sh`)

The universal bash script [`run.sh`](run.sh) automatically inspects system hardware (CPU/GPU, VRAM, free storage), configures the Python virtual environment, installs dependencies, and runs the designated pipeline mode.

#### Mode 1A: Fast Verification Mode (Default, < 30 Seconds)
Reproduces all paper tables (Tables I, II, III), generates bit-for-bit identical figures (Figures 1, 2, 3, 4), computes all statistical tests (TOST, paired $t$-test, sign test, Wilson CIs, deflated $z$-tests), and executes the 113-claim audit suite using pre-computed inference checkpoints:
```bash
# Default execution (runs fast verification automatically)
./run.sh

# Or explicitly via flag:
./run.sh --quick
```

#### Mode 1B: Full Retraining Mode (From Scratch)
Trains all models from raw InsectSet459 audio:
1. Trains the 20.3M EfficientNetV2-S oracle ceiling classifier (50 epochs, cosine schedule).
2. Trains all four generative models (OT-CFM with 640-d taxonomic FiLM, DDPM, CVAE, ACGAN) across their learning rate sweeps.
3. Generates 22,950 conditional spectrograms via 50-step Euler ODE numerical integration.
4. Executes the complete downstream classification sweep across $N=20$ independent seeds.
```bash
./run.sh --full
# Or equivalently:
./run.sh --train
```
> **Hardware Requirement:** NVIDIA GPU with $\ge 24$ GB VRAM (e.g., A40, RTX 3090, RTX 4090, A100).

#### Mode 1C: Granular Single-Step Execution
Any individual phase of the research workflow can be triggered using the `--step <N>` flag:

| Command | Target Pipeline Stage | Output Produced |
| :--- | :--- | :--- |
| `./run.sh --step 1` | **Dataset Setup & Split Audit** | Validates 60/20/20 recording split, chunk tiers, and audio specs |
| `./run.sh --step 2` | **Oracle Classifier Ceiling** | Evaluates real-data Top-1/Top-5 ceiling (`results/oracle_evaluation.json`) |
| `./run.sh --step 3` | **Generative Architecture Suite** | Verifies model parameter counts and samples test spectrograms |
| `./run.sh --step 4` | **SIPR, MMD², GCR & Collapse** | Generates **Table II** (`results/tables/table2_sipr_tiers.json`) & deflated $z$-test |
| `./run.sh --step 5` | **Cross-Architecture Replication** | Generates **Table I** (`results/tables/table1_cross_arch.json`) across LR sweeps |
| `./run.sh --step 6` | **Confidence Triage Analysis** | Quantifies 63.1% discard rate & 35 abandoned tail species |
| `./run.sh --step 7` | **Downstream $N=20$ & TOST** | Generates **Table III** (`results/tables/table3_tier4_f1.json`) & equivalence stats |
| `./run.sh --step 8` | **Render Paper Figures** | Renders all 4 publication figures into `results/figures/` |
| `./run.sh --step 9` | **Automated Claims Audit Suite** | Validates all 113 claims directly against `main.tex` |

---

### Method 2: Direct Standalone Python Execution

Researchers who prefer working directly in Python without shell scripts can activate their virtual environment and invoke any stage as an independent script:

```bash
# Activate your environment (e.g., venv or conda)
source .venv/bin/activate

# 1. Inspect splits, taxonomy, and quartile chunk counts
python 01_dataset_setup.py

# 2. Evaluate the frozen real-data oracle classifier
python 02_oracle_eval.py

# 3. Test conditional generation with OT-CFM, DDPM, CVAE, and ACGAN
python 03_generative_models.py

# 4. Measure SIPR, MMD², Genus Confusion Rate, and reproduce Table II
python 04_measure_sipr.py

# 5. Replicate cross-architecture LR sweeps and reproduce Table I
python 05_cross_arch_eval.py

# 6. Run oracle confidence triage analysis
python 06_triage_analysis.py

# 7. Evaluate N=20 paired seeds, run TOST equivalence, and reproduce Table III
python 07_downstream_and_tost.py

# 8. Render all four high-resolution figures into results/figures/
python 08_render_figures.py

# 9. Run the full 113-point claims audit suite
python 09_audit_paper_claims.py
```

---

### Method 3: Reproducing Specific Results On Demand

If you only need to inspect or replicate a particular result from the manuscript:

- **To reproduce Table I (Cross-Architecture Baselines):**
  ```bash
  python 05_cross_arch_eval.py
  # Output: results/tables/table1_cross_arch.json
  ```
- **To reproduce Table II (Tier-wise MMD², SIPR, GCR, & Wilson CIs):**
  ```bash
  python 04_measure_sipr.py
  # Output: results/tables/table2_sipr_tiers.json
  ```
- **To reproduce Table III (Downstream F1, TOST Equivalence, & ResNet50d):**
  ```bash
  python 07_downstream_and_tost.py
  # Output: results/tables/table3_tier4_f1.json
  ```
- **To reproduce all Figures (Figs 1, 2, 3, 4):**
  ```bash
  python 08_render_figures.py
  # Output: results/figures/ (fig1_longtail.png, fig2_spectrogram_grid.png, etc.)
  ```
- **To run the 113-Point Claims Verification:**
  ```bash
  python 09_audit_paper_claims.py
  # Output: results/audit_report.txt
  ```

---

### Environment Setup Alternatives

The repository is self-contained. If setting up a fresh environment manually:

```bash
# Option A: Standard Python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option B: Ultra-fast uv package manager
uv venv .venv
uv pip install -r requirements.txt

# Option C: Conda / Mamba
conda create -n bioacoustics python=3.11 -y
conda activate bioacoustics
pip install -r requirements.txt
```

---

## Pre-Trained Checkpoints & Model Architecture

The repository provides standalone inference weights for all models evaluated in the paper:

| Model Architecture | File Path | Parameter Count | Resolution / Input | Description |
| :--- | :--- | :---: | :---: | :--- |
| **EfficientNetV2-S** (Oracle) | `checkpoints/oracle/best_oracle.pt` | 20.3M | $1 \times 128 \times 235$ | Real-data ceiling classifier trained on real InsectSet459 recordings |
| **OT-CFM** (Ours) | `checkpoints/ot_cfm/best_ot_cfm.pt` | 6.97M | $1 \times 128 \times 235$ | Optimal Transport Conditional Flow Matcher with 640-d taxonomic FiLM |
| **CVAE** | `checkpoints/cvae/best_cvae.pt` | 16.18M | $1 \times 128 \times 235$ | Convolutional Variational Autoencoder baseline |
| **ACGAN** | `checkpoints/acgan/best_acgan.pt` | 8.12M | $1 \times 128 \times 235$ | Auxiliary Classifier Generative Adversarial Network baseline |
| **DDPM** | `checkpoints/ddpm/best_ddpm.pt` | 0.97M | $1 \times 128 \times 235$ | Conditional Denoising Diffusion Probabilistic Model baseline |
| **Downstream Seed Runs** | `checkpoints/phase3_chunks/*.json` | 140 runs | Metrics / F1 | Full $N=20$ seed evaluation runs across all augmentation conditions |

---

## Pipeline Architecture & Module Description

The codebase is organized into nine sequential, standalone modules:

```
IEEE SPL/
├── run.sh                          # Universal master runner
├── requirements.txt                # Pinned dependencies
├── environment.json                # Execution environment metadata
├── 01_dataset_setup.py             # Step 1: Dataset acquisition, split audit, & tier partition
├── 02_oracle_eval.py               # Step 2: Real-data oracle ceiling evaluation
├── 03_generative_models.py         # Step 3: Generative model loading & conditional sampling
├── 04_measure_sipr.py              # Step 4: SIPR, MMD², GCR & identity collapse (Table II)
├── 05_cross_arch_eval.py           # Step 5: Cross-architecture LR sweep replication (Table I)
├── 06_triage_analysis.py           # Step 6: Oracle confidence triage & tail starvation audit
├── 07_downstream_and_tost.py       # Step 7: N=20 downstream Macro-F1 & TOST equivalence (Table III)
├── 08_render_figures.py            # Step 8: Publication-quality figure generation (Figs 1–4)
├── 09_audit_paper_claims.py        # Step 9: Programmatic verification suite (113 claims)
├── src/                            # Research library
│   ├── config.py                   # Global constants, taxonomy, and paths
│   ├── dataset.py                  # Dataloaders, wideband spectrogram transforms
│   ├── models/                     # PyTorch model definitions (Classifier & Generators)
│   ├── metrics.py                  # Metric implementations (SIPR, GCR, MMD², F1)
│   └── stats.py                    # TOST, paired t-test, sign test, Wilson CIs, deflated z-test
├── checkpoints/                    # Model weights and statistical artifacts
├── data/                           # Split annotations, header metadata, species mapping
└── results/                        # Generated output tables, figures, and audit report
```

### Detailed Pipeline Steps

- **Step 1: Dataset Setup & Split Audit (`01_dataset_setup.py`):**
  Verifies the official 60/20/20 recording-level split on InsectSet459 (zero recording overlap) and defines chunk-quartile tiers: Tier 1 ($>600$ chunks, 114 species), Tier 2 ($251–600$, 115 species), Tier 3 ($119–250$, 115 species), Tier 4 ($\le 118$, 115 species).
- **Step 2: Oracle Classifier Ceiling (`02_oracle_eval.py`):**
  Evaluates the frozen real-data-trained EfficientNetV2-S oracle classifier: establishing the 48.6% Top-1 Tier-4 ceiling and 68.2% all-class ceiling.
- **Step 3: Generative Architecture Suite (`03_generative_models.py`):**
  Verifies model parameters, executes Euler ODE integration for OT-CFM ($w=2.0$ CFG), reverse diffusion sampling for DDPM, and latent decoding for CVAE and ACGAN.
- **Step 4: SIPR, MMD², GCR & Identity Collapse (`04_measure_sipr.py`):**
  Computes Table II metrics across all quartile tiers, calculates 95% Wilson score confidence intervals, verifies the 30.0 pp collapse threshold (Eq. 4), and executes the cluster-deflated $z$-test for genus confusion ($z=2.89, p=0.004$).
- **Step 5: Cross-Architecture Replication (`05_cross_arch_eval.py`):**
  Reproduces Table I across 12 generator configurations and learning-rate sweeps ($2\times 10^{-4}, 1\times 10^{-3}, 5\times 10^{-5}$), confirming identity collapse across GAN, VAE, DDPM, and Flow Matching families.
- **Step 6: Confidence Triage Analysis (`06_triage_analysis.py`):**
  Applies oracle top-1 confidence filtering, quantifying that 63.1% of synthetic samples are discarded, leaving 35/115 tail species with zero samples and 59 with $<5$, collapsing effective $K_{\text{gen}}$ from 50 to 2.83.
- **Step 7: Downstream Augmentation & TOST Equivalence (`07_downstream_and_tost.py`):**
  Aggregates $N=20$ independent paired seeds for Table III: Raw ($0.5047 \pm 0.0105$), Unfiltered Gen ($0.4961 \pm 0.0143$), Filtered Gen ($0.5029 \pm 0.0097$), and SpecAugment ($0.5174 \pm 0.0115$). Computes TOST 90% CI ($[-0.0202, -0.0088]$), paired $t$-test ($p=0.0003, d=0.98$), and sign test (16/20 wins, $p=0.012$). Replicates monotonic degradation on ResNet50d ($0.521 \to 0.512 \to 0.481$).
- **Step 8: High-Resolution Figure Rendering (`08_render_figures.py`):**
  Renders publication figures matching the paper bit-for-bit:
  - `fig1_longtail.png`: F1 vs. training chunk count for 459 species with quartile tiers and OLS trendline ($R^2=0.05, \rho=0.33$).
  - `fig2_spectrogram_grid.png`: Real vs. unfiltered synthetic spectrogram pairs for Tier-4 species.
  - `fig_baseline_spectrograms.png`: Cross-architecture synthetic spectrogram grid ($4 \times 4$).
  - `fig_degradation_curve_n20.png`: Tier-4 Macro-F1 degradation curve vs. $K_{\text{aug}}$ ($1, 5, 10, 20$) with $\pm 1\sigma$ band across $N=20$ seeds.
- **Step 9: Programmatic Claims Audit Suite (`09_audit_paper_claims.py`):**
  Executes an automated verification script testing 113 claims directly against `main.tex`, saving the complete verification matrix to `results/audit_report.txt`.

---

## Automated Paper Claims Audit

Running `./run.sh --step 9` programmatically verifies every numerical claim, table cell, confidence interval, and test in the paper:

```text
================================================================================
 STEP 9: PROGRAMMATIC VERIFICATION & CLAIMS AUDIT SUITE
 Paper: 'When Fidelity Lies: Identity Collapse in Generative Augmentation...'
================================================================================

--- AUDITING TABLE II (Tier-wise SIPR, MMD^2, GCR, & Wilson CIs) ---
 [PASS] Table II: Tier 1 Synth Chunks Evaluated            | Claimed: 3650         | Observed: 3650
 [PASS] Table II: Tier 1 MMD^2                            | Claimed: 0.047        | Observed: 0.047
 [PASS] Table II: Tier 1 SIPR (%)                         | Claimed: 64.5         | Observed: 64.5
 [PASS] Table II: Tier 1 SIPR 95% CI Lower                | Claimed: 63.0         | Observed: 63.0
 [PASS] Table II: Tier 1 SIPR 95% CI Upper                | Claimed: 66.0         | Observed: 66.0
 [PASS] Table II: Tier 1 GCR (%)                          | Claimed: 7.6          | Observed: 7.6
 [PASS] Table II: Tier 4 MMD^2                            | Claimed: 0.059        | Observed: 0.059
 [PASS] Table II: Tier 4 SIPR (%)                         | Claimed: 18.6         | Observed: 18.6
 [PASS] Table II: Tier 4 GCR (%)                          | Claimed: 12.2         | Observed: 12.2

--- AUDITING TABLE I (Cross-Architecture LR Sweeps) ---
 [PASS] Table I: ACGAN 2e-4 FAD                           | Claimed: 11.40        | Observed: 11.40
 [PASS] Table I: ACGAN 2e-4 T4 Top-1                      | Claimed: 0.0%         | Observed: 0.0%
 [PASS] Table I: CVAE Best 3e-4 FAD                       | Claimed: 8.95         | Observed: 8.95
 [PASS] Table I: CVAE Best 3e-4 T4 Top-1                  | Claimed: 6.7%         | Observed: 6.7%
 [PASS] Table I: DDPM 2e-4 FAD                            | Claimed: 4.12         | Observed: 4.12
 [PASS] Table I: DDPM 2e-4 T4 Top-1                       | Claimed: 0.0%         | Observed: 0.0%
 [PASS] Table I: OT-CFM single-LR FAD                     | Claimed: 2.18         | Observed: 2.18
 [PASS] Table I: OT-CFM single-LR T4 Top-1                | Claimed: 17.5%        | Observed: 17.5%
 [PASS] Table I: Real Oracle T4 Top-1                     | Claimed: 48.6%        | Observed: 48.6%

--- AUDITING TABLE III (N=20 Downstream & TOST Equivalence) ---
 [PASS] Table III: Raw T4 F1 Mean                         | Claimed: 0.5047       | Observed: 0.5047
 [PASS] Table III: Filtered Gen K=5 Mean                  | Claimed: 0.5029       | Observed: 0.5029
 [PASS] Table III: SpecAugment Mean                       | Claimed: 0.5174       | Observed: 0.5174
 [PASS] TOST 90% CI Lower Bound                           | Claimed: -0.0202      | Observed: -0.0202
 [PASS] TOST 90% CI Upper Bound                           | Claimed: -0.0088      | Observed: -0.0088
 [PASS] Paired t-test p-value                             | Claimed: 0.0003       | Observed: 0.0003
 [PASS] Paired t-test Cohen's d                           | Claimed: 0.98         | Observed: 0.9825
 [PASS] Binomial Sign Test SpecAugment Wins               | Claimed: 16           | Observed: 16
 [PASS] ResNet50d Secondary Raw Baseline F1               | Claimed: 0.521        | Observed: 0.521
 [PASS] ResNet50d Secondary Gen K=50 F1                   | Claimed: 0.481        | Observed: 0.481

--- AUDITING SECTION IV STATISTICAL CLAIMS ---
 [PASS] Triage: Total Synthetic Cache Samples             | Claimed: 22950        | Observed: 22950
 [PASS] Triage: Retained Surviving Cache Fraction (%)     | Claimed: 36.9         | Observed: 36.9
 [PASS] Triage: Discarded Cache Fraction (%)              | Claimed: 63.1         | Observed: 63.1
 [PASS] Triage: Abandoned Tail Species (0 samples)        | Claimed: 35           | Observed: 35
 [PASS] Triage: Starved Tail Species (< 5 samples)        | Claimed: 59           | Observed: 59
 [PASS] Correlation: Spearman rho (Fidelity vs SIPR)      | Claimed: -0.026       | Observed: -0.026
 [PASS] Identity Collapse: Observed Collapse Gap (pp)     | Claimed: 30.0         | Observed: 30.0
 [PASS] Chunk ICC: Tier 4 Intraclass Correlation          | Claimed: 0.698        | Observed: 0.6980
 [PASS] Chunk ICC: Design Effect (DEFF)                   | Claimed: 6.0          | Observed: 6.0072

--- AUDITING GENUS-LEVEL TAXONOMY & ACOUSTIC SHARING ---
 [PASS] Genus Audit: Multi-species genera count           | Claimed: 74           | Observed: 74
 [PASS] Genus Audit: Within-genus cosine distance         | Claimed: 0.174        | Observed: 0.1740
 [PASS] Genus Audit: Cross-genus cosine distance          | Claimed: 0.182        | Observed: 0.1823
 [PASS] Genus Audit: Distance reduction percentage (%)    | Claimed: 4.55         | Observed: 4.5549
 [PASS] Genus Audit: Acoustic difference p-value          | Claimed: 0.006        | Observed: 0.0064
 [PASS] Genus Confusion: Real data error rate (%)         | Claimed: 8.1          | Observed: 8.1
 [PASS] Genus Confusion: Synthetic error rate (%)         | Claimed: 12.2         | Observed: 12.2
 [PASS] Genus Confusion: Standard z-statistic             | Claimed: 5.05         | Observed: 5.0435
 [PASS] Genus Confusion: Cluster-deflated z-stat          | Claimed: 2.89         | Observed: 2.8926
 [PASS] Genus Confusion: Cluster-deflated p-value         | Claimed: 0.004        | Observed: 0.0038

--- AUDITING RENDERED FIGURE ARTIFACTS ---
 [PASS] Figure artifact: fig1_longtail.png                | Claimed: True         | Observed: True
 [PASS] Figure artifact: fig2_spectrogram_grid.png        | Claimed: True         | Observed: True
 [PASS] Figure artifact: fig_baseline_spectrograms.png    | Claimed: True         | Observed: True
 [PASS] Figure artifact: fig_degradation_curve_n20.png    | Claimed: True         | Observed: True

================================================================================
 FINAL AUDIT RESULT: 113 / 113 CLAIMS VERIFIED (100.0%)
 [SUCCESS] 100% OF PAPER CLAIMS REPRODUCED PERFECTLY!
================================================================================
```

---

## Citation

```bibtex
@article{krishna2026fidelity,
  author    = {Krishna, Nived and Kala, S},
  title     = {When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale},
  journal   = {IEEE Signal Processing Letters (Under Publication)},
  year      = {2026},
  url       = {https://github.com/NivedKris/insectset459-generative}
}
```

---

## License

This research codebase is licensed under the [MIT License](LICENSE). The InsectSet459 dataset is distributed under its original licensing terms as specified in its [Zenodo release (Record 14056458)](https://zenodo.org/records/14056458).
