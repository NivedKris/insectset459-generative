# IEEE Signal Processing Letters (SPL) Reproducibility Kit

Official reproducibility repository for the manuscript submitted to the *IEEE Signal Processing Letters*:

> **When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale**  
> **Authors:** Nived Krishna$^1$ and Kala S$^{2,*}$, *Senior Member, IEEE*  
> $^1$*Department of Computer Science & Engineering, Indian Institute of Technology Kharagpur, India*  
> $^2$*Department of Electronics & Communication Engineering, Indian Institute of Information Technology Kottayam, India*  

---

## 🚀 Quick Start (< 3 Minutes)

To verify all claims, tables (Tables I, II, III), figures (Figures 1, 2, 3, 4), and statistical audits:

```bash
# Clone the repository
git clone https://github.com/NivedKris/insectset459-generative.git
cd "IEEE SPL"

# Execute the universal master script in quick reproduction mode
bash run.sh --quick
```

All figures are rendered into `results/figures/` and all numerical tables into `results/tables/`.

---

## 📋 Execution Modes

The reproducibility kit provides two operational modes via `run.sh`:

1. **Fast Mode (`./run.sh --quick` or `./run.sh`) [Default]**:
   - Uses pre-trained inference checkpoints (all strictly `< 82 MB`, conforming to GitHub's 100 MB limit) and evaluation logs.
   - Computes exact Wilson score confidence intervals, Spearman rank correlations, deflated $z$-tests, Two One-Sided Tests (TOST) equivalence, and paired $t$-tests.
   - Renders high-resolution publication figures and runs an automated 44-point verification audit against `main.tex`.
   - **Runtime**: $< 3$ minutes on any modern workstation with GPU or CPU.

2. **Full Training Mode (`./run.sh --full` or `--train`)**:
   - Trains the EfficientNetV2-S oracle classifier from scratch on InsectSet459.
   - Trains the 4 generative families (OT-CFM, DDPM, CVAE, ACGAN).
   - Generates conditional spectrograms with Euler ODE integration and CFG ($w=2.0$).
   - Executes the downstream $N=20$ paired seed classification sweep.
   - **Hardware Recommended**: NVIDIA GPU with $\ge 24$ GB VRAM (e.g., A40, RTX 3090, RTX 4090, A100).

---

## 📦 GitHub 100 MB Limit Compliance

GitHub blocks pushes containing files larger than 100 MB (`GH001`). All model weights in this kit have been audited and stripped of non-inference training optimizer states:

| Model Architecture | Checkpoint File | Parameter Count | Disk Size | GitHub Safe? |
| :--- | :--- | :--- | :--- | :--- |
| **Oracle Classifier** (EfficientNetV2-S) | `checkpoints/oracle/best_oracle.pt` | 20.3M | **80.07 MB** | ✅ `< 100 MB` |
| **OT-CFM Generator** (Ours, Flow Matching) | `checkpoints/ot_cfm/best_ot_cfm.pt` | 6.97M | **26.65 MB** | ✅ `< 100 MB` |
| **CVAE Generator** (Variational Autoencoder) | `checkpoints/cvae/best_cvae.pt` | 16.18M | **62.00 MB** | ✅ `< 100 MB` |
| **ACGAN Generator** (Adversarial GAN) | `checkpoints/acgan/best_acgan.pt` | 8.12M | **31.00 MB** | ✅ `< 100 MB` |
| **DDPM Generator** (Diffusion Model) | `checkpoints/ddpm/best_ddpm.pt` | 0.97M | **3.80 MB** | ✅ `< 100 MB` |
| **Downstream Evaluation Chunks** | `checkpoints/phase3_chunks/*.json` | 140 runs | **< 2.0 MB** | ✅ `< 100 MB` |

*Note*: Large raw audio archives (51 GB) and multi-GB floating-point caches are excluded via `.gitignore`. If local datasets are present, the kit automatically links them; otherwise, it downloads official annotations and audio from Zenodo.

---

## 🔬 Step-by-Step Module Architecture

Each phase is accessible as a modular, standalone Python script or through `run.sh --step <1-9>`:

### Step 1: Dataset Setup & Split Insight Audit (`01_dataset_setup.py`)
- Automatically links local InsectSet459 audio or downloads from Zenodo Record `14056458`.
- Verifies zero recording-ID overlap across Train (15,788), Validation (5,290), and Test (5,219) splits.
- Computes quartile tiers by training chunk counts:
  - **Tier 1 (Head)**: $> 600$ chunks ($n=114$ species)
  - **Tier 2 (Mid)**: $251 - 600$ chunks ($n=115$ species)
  - **Tier 3 (Few)**: $119 - 250$ chunks ($n=115$ species)
  - **Tier 4 (Tail)**: $\le 118$ chunks ($n=115$ species, minimum 4 test chunks)
- Validates 48 kHz wideband log-mel spectrogram extraction: $(1 \times 128 \times 235)$.

### Step 2: Oracle Classifier & Real-Data Ceiling (`02_oracle_eval.py`)
- Loads the real-data-trained EfficientNetV2-S oracle.
- Measures oracle performance on real test chunks:
  - **Tier-4 Real Accuracy**: **48.6% Top-1** (76.5% Top-5)
  - **All-Class Real Accuracy**: **68.2% Top-1** (87.4% Top-5)
  - Sets the real-data ceiling $\overline{\text{SIPR}}^{\text{real}}_T$.

### Step 3: Generative Architecture Suite (`03_generative_models.py`)
- Loads and verifies parameter counts across all 4 generator families.
- Generates conditional test spectrograms using Euler ODE flow matching ($w=2.0$), reverse diffusion, and latent decoding.

### Step 4: SIPR, MMD², Genus Confusion & Identity Collapse (`04_measure_sipr.py`)
- Reproduces **Table II** of the manuscript:
  - **Tier 1**: $\text{MMD}^2 = 0.047$, $\text{SIPR} = 64.5\%$ [63.0, 66.0], $\text{GCR} = 7.6\%$ [6.8, 8.5]
  - **Tier 2**: $\text{MMD}^2 = 0.052$, $\text{SIPR} = 49.2\%$ [48.0, 50.4], $\text{GCR} = 7.4\%$ [6.8, 8.1]
  - **Tier 3**: $\text{MMD}^2 = 0.057$, $\text{SIPR} = 26.8\%$ [25.8, 27.8], $\text{GCR} = 9.0\%$ [8.4, 9.7]
  - **Tier 4**: $\text{MMD}^2 = 0.059$, $\text{SIPR} = 18.6\%$ [17.6, 19.6], $\text{GCR} = 12.2\%$ [11.4, 13.1]
- **Identity Collapse (Eq. 4)**: $48.6\% - 18.6\% = 30.0\text{ pp} \ge 30.0\text{ pp}$ margin (CONFIRMED).
- **Fidelity-Identity Divergence**: Spearman $\rho = -0.026$ ($p = 0.585$), proving distributional fidelity metrics (MMD$^2$/FAD) cannot certify per-sample identity.
- **Cluster-Deflated $z$-test** ($\text{DEFF} = 6.0$, $\text{ICC} = 0.698$): Genus confusion gap ($z = 2.89, p = 0.004$) is statistically real but far too small (4.1 pp) to explain the 30.0 pp collapse.

### Step 5: Cross-Architecture Replication (`05_cross_arch_eval.py`)
- Reproduces **Table I** of the manuscript across learning-rate sweeps ($2\times 10^{-4}$, $1\times 10^{-3}$, $5\times 10^{-5}$):
  - **ACGAN (8.1M)**: Collapses to $0.0\%$ Tier-4 Top-1 across all LRs.
  - **DDPM (1.0M)**: Collapses to $0.0\%$ Tier-4 Top-1 across all LRs.
  - **CVAE (16.2M)**: Peaks at $6.7\%$ Tier-4 Top-1 (Best LR $3\times 10^{-4}$).
  - **OT-CFM (Ours)**: Reaches **17.5%** (single-LR) / **18.6%** (two-stage), outperforming all baselines ($p < 0.001$).
  - Proves identity collapse is architecture-general.

### Step 6: Oracle Triage Analysis (`06_triage_analysis.py`)
- Implements confidence-based oracle filtering ($\text{argmax}_k f_\phi(x)_k = c$).
- Quantifies the severe triage effect on tail species:
  - $63.1\%$ of the 22,950-sample cache discarded ($36.9\%$ retained).
  - **35 out of 115 tail species** left with ZERO surviving synthetic samples.
  - **59 additional tail species** left with $<5$ samples.
  - Effective mean $K_{\text{gen}}$ in Tier 4 collapses from 50 to 2.83.
  - Paired $t$-test (Filtered vs. Raw): $t = -0.51, p = 0.61, d = -0.11$ (statistically indistinguishable from no augmentation).

### Step 7: Downstream Augmentation & $N=20$ TOST Equivalence (`07_downstream_and_tost.py`)
- Evaluates 20 independent paired seeds and reproduces **Table III**:
  - **Raw**: $0.5047 \pm 0.0105$
  - **Unfiltered Generative ($K=5$)**: $0.4961 \pm 0.0143$
  - **Filtered Generative ($K=5$)**: $0.5029 \pm 0.0097$
  - **SpecAugment**: $0.5174 \pm 0.0115$
- **Pre-Registered TOST Equivalence ($\pm 0.020$ margin)**:
  - 90% CI: $[-0.0202, -0.0088]$, $p = 0.055$ (**Inconclusive**).
- **Paired $t$-test**: $t = -4.394, p = 0.0003, d = 0.98$ (SpecAugment significantly outperforms filtered generative augmentation).
- **Binomial Sign Test**: 16/20 wins for SpecAugment ($p = 0.012$).
- **Secondary Architecture Audit**: ResNet50d ($N=3$ seeds) replicates monotonic degradation (Raw 0.521, Gen $K=5$: 0.512, Gen $K=50$: 0.481).

### Step 8: Rendering High-Resolution Paper Figures (`08_render_figures.py`)
- Generates publication-quality images:
  - `fig1_longtail.png`: Long-tail collapse vs training chunks with quartile colors and OLS trendline ($R^2=0.05, \rho=0.33$).
  - `fig2_spectrogram_grid.png`: Real vs unfiltered synthetic spectrograms for Tier-4 species.
  - `fig_baseline_spectrograms.png`: Cross-architecture grid ($4 \times 4$) comparing ACGAN, CVAE, OT-CFM, and DDPM.
  - `fig_degradation_curve_n20.png`: Dual-panel figure: Left = MMD$^2$ distribution ($0.058 \pm 0.019$); Right = Tier-4 Macro-F1 degradation curve vs $K_{\text{aug}}$ ($1, 5, 10, 20$) with $\pm 1\sigma$ band across 20 seeds.

### Step 9: Claims & Numerical Audit Suite (`09_audit_paper_claims.py`)
- Automatically validates **44 out of 44 claims and numerical values** from `main.tex`.
- Generates `results/audit_report.txt` with a 100% pass rate.

---

## 📊 Summary of Main Results

```
================================================================================
FINAL VERIFICATION AUDIT (44/44 CLAIMS VERIFIED - 100% AGREEMENT WITH MAIN.TEX)
================================================================================
[PASS] Table II: Tier 1 MMD^2               : 0.047
[PASS] Table II: Tier 1 SIPR (%)            : 64.5% [63.0, 66.0]
[PASS] Table II: Tier 4 MMD^2               : 0.059
[PASS] Table II: Tier 4 SIPR (%)            : 18.6% [17.6, 19.6]
[PASS] Table II: Tier 4 GCR (%)             : 12.2% [11.4, 13.1]
[PASS] Identity Collapse Equation 4 (Gap)   : 30.0 pp (Exceeds >= 30.0 pp)
[PASS] Fidelity-Identity Spearman rho       : -0.026 (p = 0.585, no correlation)
[PASS] Table I: CVAE Best Tier-4 Top-1      : 6.7% (FAD 8.95)
[PASS] Table I: OT-CFM Single-LR T4 Top-1   : 17.5% (FAD 2.18)
[PASS] Table I: Real Data Oracle T4 Top-1   : 48.6% (Top-5: 76.5%)
[PASS] Triage: Tail Species Abandoned (0)   : 35 / 115 species
[PASS] Triage: Tail Species Starved (< 5)   : 59 / 115 species
[PASS] Triage: Effective Mean K_gen in Tail : 2.83 samples
[PASS] Table III: Raw Tier-4 Macro-F1       : 0.5047 ± 0.0105
[PASS] Table III: Filtered Gen K=5 Macro-F1 : 0.5029 ± 0.0097
[PASS] Table III: SpecAugment Macro-F1      : 0.5174 ± 0.0115
[PASS] TOST 90% Confidence Interval         : [-0.0202, -0.0088] (p = 0.055, Inconclusive)
[PASS] Paired t-test (SpecAug vs Filtered)  : p = 0.0003, Cohen's d = 0.98
[PASS] Binomial Sign Test                   : 16/20 wins for SpecAugment (p = 0.012)
[PASS] Secondary Classifier (ResNet50d)     : Raw 0.521 -> K=5 0.512 -> K=50 0.481
================================================================================
```

---

## 📂 Repository Directory Layout

```
IEEE SPL/
├── run.sh                          # Universal master entrypoint
├── README.md                       # Comprehensive documentation
├── requirements.txt                # Pinned Python dependencies
├── environment.json                # Hardware & environment specification
├── .gitignore                      # GitHub file size safety rules (<100MB)
├── 01_dataset_setup.py             # Step 1: Data acquisition & tier audit
├── 02_oracle_eval.py               # Step 2: Base oracle real ceiling
├── 03_generative_models.py         # Step 3: Generative models suite & sampling
├── 04_measure_sipr.py              # Step 4: SIPR, MMD², GCR & collapse (Table II)
├── 05_cross_arch_eval.py           # Step 5: Cross-architecture replication (Table I)
├── 06_triage_analysis.py           # Step 6: Triage & 35-abandoned tail audit
├── 07_downstream_and_tost.py       # Step 7: Downstream F1 & N=20 TOST (Table III)
├── 08_render_figures.py            # Step 8: High-resolution figure renderer
├── 09_audit_paper_claims.py        # Step 9: Programmatic verification suite
├── src/                            # Reusable library package
│   ├── config.py                   # Constants, tier bounds, paths
│   ├── dataset.py                  # Chunk-level dataloaders & transforms
│   ├── models/                     # Classifier & Generator definitions
│   ├── metrics.py                  # SIPR, GCR, MMD², Macro-F1
│   └── stats.py                    # TOST, paired t-test, sign test, Wilson CI
├── checkpoints/                    # ALL FILES STRICTLY < 82 MB
│   ├── oracle/best_oracle.pt       # EfficientNetV2-S weights (80.07 MB)
│   ├── ot_cfm/best_ot_cfm.pt       # OT-CFM EMA weights (26.65 MB)
│   ├── cvae/best_cvae.pt           # CVAE baseline weights (62.00 MB)
│   ├── acgan/best_acgan.pt         # ACGAN baseline weights (31.00 MB)
│   ├── ddpm/best_ddpm.pt           # DDPM baseline weights (3.80 MB)
│   └── phase3_chunks/              # All 140 N=20 run chunks (< 2.0 MB)
└── results/                        # Generated reproducibility outputs
    ├── tables/                     # Tables I, II, III in JSON and formatted text
    ├── figures/                    # High-res Figures 1, 2, 3, 4
    └── audit_report.txt            # Complete claims verification matrix
```

---

## 📜 Citation

If you use this reproducibility kit or build upon our findings, please cite:

```bibtex
@article{krishna2026fidelity,
  author    = {Krishna, Nived and Kala, S},
  title     = {When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale},
  journal   = {IEEE Signal Processing Letters},
  year      = {2026},
  note      = {Code: https://github.com/NivedKris/insectset459-generative}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The InsectSet459 dataset is subject to its original terms and licenses from its Zenodo publication (Record 14056458).
