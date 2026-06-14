#!/bin/bash
set -euo pipefail

# Attribution + MoDISco + FIMO enrichment pipeline for one model × cell type × event type.
#
# This script orchestrates four steps:
#   05_attribution_prep.py  — filter input_dedup.csv and add PSI bins  (local)
#   01_get_attributions.py  — integrated-gradients attributions         (eLife repo)
#   02_run_modisco.py       — de novo motif discovery                    (eLife repo)
#   03_run_enrichment.py    — FIMO enrichment comparison                 (eLife repo)
#
# Steps 01–03 are delegated to a local clone of the eLife companion repo
# (https://github.com/jsdearbo/elife_delay_splice). Pass its path with
# --elife-repo or set ELIFE_REPO in your environment.
#
# Usage:
#   bash 06_attribution.sh \
#       --run-name B_cell_SE \
#       --config   ../manuscript_runs/configs/B_cell_SE.yaml \
#       --input-csv /path/to/SE_results/input_dedup.csv \
#       --elife-repo /path/to/elife_delay_splice \
#       [--out-dir ./attribution_runs] \
#       [--cuda-device 0] \
#       [--psi-thresholds 0.5 0.95] \
#       [--skip-prepare] [--skip-attr] [--skip-modisco] [--skip-enrichment]
#
# --psi-thresholds is recommended for SE events whose PSI is skewed toward 1.0.
# Omit it to use balanced tertile quantiles (default, works well for RI events).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
UTILS_DIR="${REPO_ROOT}/utils"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_NAME=""
CONFIG=""
INPUT_CSV=""
OUT_DIR="${REPO_ROOT}/attribution_runs"
ELIFE_REPO="${ELIFE_REPO:-}"
CUDA_DEVICE=0
EVENTS_PER_BIN=100
PSI_THRESHOLDS=""
PSI_BIN_FILTER=""
SKIP_PREPARE=0
SKIP_ATTR=0
SKIP_MODISCO=0
SKIP_ENRICHMENT=0
RUN_MOTIF_MAPS=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)           RUN_NAME="$2";           shift 2 ;;
        --config)             CONFIG="$2";              shift 2 ;;
        --input-csv)          INPUT_CSV="$2";           shift 2 ;;
        --out-dir)            OUT_DIR="$2";             shift 2 ;;
        --elife-repo)         ELIFE_REPO="$2";          shift 2 ;;
        --cuda-device)        CUDA_DEVICE="$2";         shift 2 ;;
        --events-per-bin)     EVENTS_PER_BIN="$2";     shift 2 ;;
        --psi-thresholds)     PSI_THRESHOLDS="$2 $3";  shift 3 ;;
        --psi-bin-filter)     PSI_BIN_FILTER="$2";     shift 2 ;;
        --skip-prepare)       SKIP_PREPARE=1;           shift   ;;
        --skip-attr)          SKIP_ATTR=1;              shift   ;;
        --skip-modisco)       SKIP_MODISCO=1;           shift   ;;
        --skip-enrichment)    SKIP_ENRICHMENT=1;        shift   ;;
        --run-motif-maps)     RUN_MOTIF_MAPS=1;         shift   ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: --run-name is required (e.g. B_cell_SE)" >&2; exit 1
fi
if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required (path to YAML config)" >&2; exit 1
fi
if [[ -z "$INPUT_CSV" ]]; then
    echo "ERROR: --input-csv is required (path to input_dedup.csv from 02_extract_embeddings step)" >&2; exit 1
fi
if [[ -z "$ELIFE_REPO" ]]; then
    echo "ERROR: --elife-repo is required (path to local clone of elife_delay_splice repo)" >&2
    echo "       Clone it with: git clone https://github.com/jsdearbo/elife_delay_splice" >&2
    exit 1
fi
if [[ ! -d "$ELIFE_REPO" ]]; then
    echo "ERROR: eLife repo not found at: $ELIFE_REPO" >&2; exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2; exit 1
fi

RUN_DIR="${OUT_DIR}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

# Derive cell type and event type from run name (e.g. B_cell_SE → B_cell, SE)
EVENT_TYPE="${RUN_NAME##*_}"
CELL_TYPE="${RUN_NAME%_*}"

# Expose this repo's utils/ and eLife's utils/ package on PYTHONPATH
export PYTHONPATH="${ELIFE_REPO}:${UTILS_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo " Attribution Pipeline: ${RUN_NAME}"
echo " config        : ${CONFIG}"
echo " run dir       : ${RUN_DIR}"
echo " input CSV     : ${INPUT_CSV}"
echo " cell type     : ${CELL_TYPE}"
echo " event type    : ${EVENT_TYPE}"
echo " eLife repo    : ${ELIFE_REPO}"
echo " cuda device   : ${CUDA_DEVICE}"
echo " events/bin    : ${EVENTS_PER_BIN}"
echo "============================================================"

# ── Step 05: Prepare events CSV ──────────────────────────────────────────────
if [[ $SKIP_PREPARE -eq 1 && -f "${RUN_DIR}/events.csv" ]]; then
    echo "[skip] Step 05 — using existing ${RUN_DIR}/events.csv"
else
    echo ""
    echo "── Step 05: Prepare events ──────────────────────────────────────────────"
    PSI_THRESH_ARGS=()
    if [[ -n "$PSI_THRESHOLDS" ]]; then
        # shellcheck disable=SC2206
        PSI_THRESH_ARGS=(--psi-thresholds $PSI_THRESHOLDS)
    fi
    PSI_BIN_FILTER_ARGS=()
    if [[ -n "$PSI_BIN_FILTER" ]]; then
        PSI_BIN_FILTER_ARGS=(--psi-bin-filter "$PSI_BIN_FILTER")
    fi
    python "${SCRIPT_DIR}/05_attribution_prep.py" \
        --input-csv      "${INPUT_CSV}" \
        --cell-type      "${CELL_TYPE}" \
        --event-type     "${EVENT_TYPE}" \
        --out-csv        "${RUN_DIR}/events.csv" \
        --events-per-bin "${EVENTS_PER_BIN}" \
        "${PSI_THRESH_ARGS[@]}" \
        "${PSI_BIN_FILTER_ARGS[@]}"
fi

# ── Step 01: Attributions (GPU) ───────────────────────────────────────────────
if [[ $SKIP_ATTR -eq 1 && -f "${RUN_DIR}/attributions.pkl" ]]; then
    echo "[skip] Step 01 — using existing attributions.pkl"
else
    echo ""
    echo "── Step 01: Attributions (GPU, device ${CUDA_DEVICE}) ───────────────────"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    python "${ELIFE_REPO}/scripts/01_get_attributions.py" \
        --config "${CONFIG}"
fi

# ── Step 02: MoDISco ──────────────────────────────────────────────────────────
if [[ $SKIP_MODISCO -eq 1 && -d "${RUN_DIR}/all_seqs_modisco" ]]; then
    echo "[skip] Step 02 — using existing modisco outputs"
else
    echo ""
    echo "── Step 02: MoDISco ─────────────────────────────────────────────────────"
    python "${ELIFE_REPO}/scripts/02_run_modisco.py" \
        --config "${CONFIG}"
fi

# ── Step 03: FIMO enrichment ──────────────────────────────────────────────────
if [[ $SKIP_ENRICHMENT -eq 1 && -d "${RUN_DIR}/enrichment" ]]; then
    echo "[skip] Step 03 — using existing enrichment outputs"
else
    echo ""
    echo "── Step 03: FIMO enrichment ─────────────────────────────────────────────"
    python "${ELIFE_REPO}/scripts/03_run_enrichment.py" \
        --config "${CONFIG}"
fi

# ── Step 04: Motif maps (optional) ───────────────────────────────────────────
if [[ $RUN_MOTIF_MAPS -eq 1 ]]; then
    echo ""
    echo "── Step 04: Motif maps ──────────────────────────────────────────────────"
    python "${ELIFE_REPO}/scripts/04_map_motifs.py" \
        --config "${CONFIG}"
fi

echo ""
echo "============================================================"
echo " Done: ${RUN_NAME}"
echo " Outputs: ${RUN_DIR}"
echo "============================================================"
