#!/bin/bash
# Test evaluation on existing checkpoint to verify everything works
source "$(dirname "$0")/vars.sh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export BASE="/work/${USERNAME}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503
# Activate the radlads2 conda environment before running anything else
source /work/parsa/miniconda3/condabin/conda activate radlads2


bsz=32
tasks=winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa,lambada_openai,gsm8k,mmlu,mathqa,race

# Test checkpoint path (removing the $ at the end if present)
TEST_CKPT_PATH="/work/hei/radlads/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-1/ckpt-final.pth"

echo "Testing evaluation on: $TEST_CKPT_PATH"

if [ -f "$TEST_CKPT_PATH" ]; then
    echo "Checkpoint exists. Running evaluation..."
    echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    # Count number of GPUs from CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
    echo "Using ${NUM_GPUS} GPUs with DistributedDataParallel via torchrun"
    torchrun --nproc-per-node=${NUM_GPUS} multi_gpu_run_lm_eval.py \
        -c ${qwen_yaml} \
        -c configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/gdn.yaml \
        --path "${TEST_CKPT_PATH}" \
        --is_pretrained no \
        --bsz 4 \
        --precision bf16 \
        --tokenizer_name ${tokenizer} \
        --tasks ${tasks}

else
    echo "Checkpoint does not exist: $TEST_CKPT_PATH"
    exit 1
fi

