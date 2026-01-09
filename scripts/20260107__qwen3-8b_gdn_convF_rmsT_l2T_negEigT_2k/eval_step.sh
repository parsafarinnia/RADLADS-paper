#!/bin/bash
# Evaluate final checkpoint for a specific step
# Usage: eval_step.sh <step_number> (0, 1, or 2)

source "$(dirname "$0")/vars.sh"

STEP_NUM=$1
if [ -z "$STEP_NUM" ]; then
    echo "Usage: $0 <step_number> (0, 1, or 2)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export BASE="/work/${USERNAME}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503

bsz=32
tasks=winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,gsm8k,mmlu,mathqa,race

case $STEP_NUM in
    0)
        CKPT_PATH="${STEP0_PTH_PATH}/ckpt-final.pth"
        ;;
    1)
        CKPT_PATH="${STEP1_PTH_PATH}/ckpt-final.pth"
        ;;
    2)
        CKPT_PATH="${STEP2_PTH_PATH}/ckpt-final.pth"
        bsz=1  # Step 2 uses smaller batch size
        ;;
    *)
        echo "Invalid step number: $STEP_NUM. Must be 0, 1, or 2"
        exit 1
        ;;
esac

if [ -f "$CKPT_PATH" ]; then
    echo "Evaluating checkpoint: $CKPT_PATH"
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python run_lm_eval.py \
        -c ${qwen_yaml} \
        -c ${la_yaml} \
        --path "${CKPT_PATH}" \
        --is_pretrained no \
        --bsz ${bsz} \
        --tokenizer_name ${tokenizer} \
        --tasks ${tasks}
else
    echo "Checkpoint does not exist: $CKPT_PATH, skipping."
    exit 1
fi

