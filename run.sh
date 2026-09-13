#!/usr/bin/env bash
# ==============================================================================
# InsectSet459 Generative Benchmark Suite
# Study: "When Fidelity Lies: Identity Collapse in Generative Augmentation at 459-Species Extreme Long-Tail Scale"
# Authors: Nived Krishna and Kala S
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="quick"
STEP="all"

print_help() {
    echo "=============================================================================="
    echo " Benchmark & Verification Suite - Master Runner"
    echo "=============================================================================="
    echo "Usage: ./run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --quick          Fast reproduction mode using pre-trained checkpoints (default)"
    echo "                   Reproduces all tables, figures, and audits in < 3 minutes."
    echo "  --full, --train  Full mode: trains models from scratch (requires >=24GB GPU)"
    echo "  --step <N>       Execute a single step (1 to 9):"
    echo "                     1: Dataset Setup & Split Insight Audit"
    echo "                     2: Oracle Classifier Real-Data Ceiling"
    echo "                     3: Generative Architecture Suite & Sampling"
    echo "                     4: SIPR, MMD², Genus Confusion & Collapse (Table II)"
    echo "                     5: Cross-Architecture LR Sweep Replication (Table I)"
    echo "                     6: Oracle Confidence Triage Analysis"
    echo "                     7: Downstream Augmentation & N=20 TOST Equivalence (Table III)"
    echo "                     8: Render All High-Resolution Figures"
    echo "                     9: Automated Claims & Numerical Audit Suite"
    echo "  --help, -h       Show this help message and exit"
    echo "=============================================================================="
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            MODE="quick"
            shift
            ;;
        --full|--train)
            MODE="full"
            shift
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo " Benchmark & Verification Pipeline"
echo " Study: When Fidelity Lies: Identity Collapse in Generative Augmentation..."
echo " Execution Mode: [${MODE^^}] | Target Step: [${STEP}]"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Pre-Flight Environment & Hardware Check
# ------------------------------------------------------------------------------
echo ""
echo "[1/4] Pre-Flight Checks:"

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || echo "NVIDIA GPU")
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n 1 || echo "")
    echo "  [OK] Detected GPU: ${GPU_NAME} (${GPU_MEM})"
else
    echo "  [INFO] Hardware: PyTorch device backend active."
fi

FREE_GB=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
echo "  [OK] Free Disk Space: ${FREE_GB} GB available."

# ------------------------------------------------------------------------------
# Virtual Environment Provisioning
# ------------------------------------------------------------------------------
echo ""
echo "[2/4] Python Environment Provisioning:"

# Check for active or local virtual environment
PYTHON_CMD="python3"
if [ -d "../.venv" ]; then
    PYTHON_CMD="../.venv/bin/python"
    echo "  [OK] Using workspace virtualenv: ${PYTHON_CMD}"
elif [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo "  [OK] Using local virtualenv: ${PYTHON_CMD}"
else
    if command -v uv &> /dev/null; then
        echo "  [INFO] Provisioning rapid virtualenv via uv..."
        uv venv .venv
        uv pip install -r requirements.txt
        PYTHON_CMD=".venv/bin/python"
    else
        echo "  [INFO] Creating virtualenv via python3 -m venv..."
        python3 -m venv .venv
        .venv/bin/pip install -q -r requirements.txt
        PYTHON_CMD=".venv/bin/python"
    fi
fi

# Ensure output directories exist
mkdir -p results/{tables,figures,measurements} data checkpoints

# ------------------------------------------------------------------------------
# Pipeline Step Execution Functions
# ------------------------------------------------------------------------------
run_step_1() {
    echo ""
    echo ">>> Running Step 1: Dataset Setup & Split Insight Analysis..."
    $PYTHON_CMD 01_dataset_setup.py --mode "$MODE"
}

run_step_2() {
    echo ""
    echo ">>> Running Step 2: Oracle Classifier & Real-Data Ceiling..."
    if [ "$MODE" = "full" ]; then
        $PYTHON_CMD 02_oracle_eval.py --mode full --train
    else
        $PYTHON_CMD 02_oracle_eval.py --mode quick
    fi
}

run_step_3() {
    echo ""
    echo ">>> Running Step 3: Generative Architecture Suite & Sampling..."
    $PYTHON_CMD 03_generative_models.py --mode "$MODE"
}

run_step_4() {
    echo ""
    echo ">>> Running Step 4: Measuring SIPR, MMD², Genus Confusion & Collapse (Table II)..."
    $PYTHON_CMD 04_measure_sipr.py --mode "$MODE"
}

run_step_5() {
    echo ""
    echo ">>> Running Step 5: Cross-Architecture Replication (Table I)..."
    $PYTHON_CMD 05_cross_arch_eval.py --mode "$MODE"
}

run_step_6() {
    echo ""
    echo ">>> Running Step 6: Oracle Confidence Triage Analysis..."
    $PYTHON_CMD 06_triage_analysis.py --mode "$MODE"
}

run_step_7() {
    echo ""
    echo ">>> Running Step 7: Downstream Classification & TOST Equivalence (Table III)..."
    $PYTHON_CMD 07_downstream_and_tost.py --mode "$MODE"
}

run_step_8() {
    echo ""
    echo ">>> Running Step 8: Rendering High-Resolution Figures..."
    $PYTHON_CMD 08_render_figures.py
}

run_step_9() {
    echo ""
    echo ">>> Running Step 9: Automated Claims & Numerical Audit Suite..."
    $PYTHON_CMD 09_audit_paper_claims.py
}

# ------------------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------------------
echo ""
echo "[3/4] Executing Pipeline Steps:"

case "$STEP" in
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    4) run_step_4 ;;
    5) run_step_5 ;;
    6) run_step_6 ;;
    7) run_step_7 ;;
    8) run_step_8 ;;
    9) run_step_9 ;;
    all)
        run_step_1
        run_step_2
        run_step_3
        run_step_4
        run_step_5
        run_step_6
        run_step_7
        run_step_8
        run_step_9
        ;;
    *)
        echo "[ERROR] Invalid step: $STEP. Valid options are 1-9 or all."
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# Final Report Summary
# ------------------------------------------------------------------------------
echo ""
echo "[4/4] Pipeline Complete! Generated Artifacts Summary:"
echo "------------------------------------------------------------------------------"
echo "  [Tables Generated in results/tables/]:"
ls -lh results/tables/
echo ""
echo "  [Figures Rendered in results/figures/]:"
ls -lh results/figures/
echo ""
echo "  [Claims Verification Report]:"
cat results/audit_report.txt | head -n 8
echo "------------------------------------------------------------------------------"
echo "=============================================================================="
echo " [SUCCESS] Benchmark Pipeline Execution Complete!"
echo "=============================================================================="
