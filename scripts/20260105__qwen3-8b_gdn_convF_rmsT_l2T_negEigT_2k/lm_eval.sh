#!/bin/bash
source "$(dirname "$0")/vars.sh"
echo $RUN_NAME
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #1,3,5 #0,1,2,3,4,5,6,7 #0,1,3,4,5,6,7
export BASE="${1:-/work/${USERNAME}}/radlads"
export HF_CACHE_DIR="${BASE}/.cache/huggingface/hub"
export MAIN_PROCESS_PORT=29503

bsz=32




tasks=winogrande,arc_easy,arc_challenge,piqa,mmlu,mathqa,race,gsm8k
# tasks=arc_challenge
# tasks=gsm8k
# # Originally pretrained ##
# ORIGINAL_MODEL_NAME="Qwen3-8B-Base"
# ORIGINAL_MODEL_PATH="Qwen/${ORIGINAL_MODEL_NAME}"
# MODEL_PATH="${BASE}/pths/${ORIGINAL_MODEL_NAME}/pretrained.pth"
# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
# python run_lm_eval.py \
#     --path ${ORIGINAL_MODEL_PATH} \
#     -c ${qwen_yaml} \
#     --is_pretrained yes \
#     --bsz ${bsz} \
#     --tasks ${tasks}
#     # --tasks winogrande,arc_easy,arc_challenge,hellaswag,piqa,openbookqa









checkpoints=()
# checkpoints+=('init')
# start=0
# end=20
# stride=1
# for i in $(seq $start $stride $end); do
#     checkpoints+=("$i")
# done
checkpoints+=('final')
# tasks=mmlu,lambada_openai,hellaswag





# for i in "${checkpoints[@]}"; do
#     CKPT_PATH="${STEP0_PTH_PATH}/ckpt-${i}.pth"
#     if [ -f "$CKPT_PATH" ]; then
#         echo "Evaluating checkpoint: $CKPT_PATH"
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         python run_lm_eval.py \
#             -c ${qwen_yaml} \
#             -c ${la_yaml} \
#             --path "${CKPT_PATH}" \
#             --is_pretrained no \
#             --bsz ${bsz} \
#             --tokenizer_name ${tokenizer} \
#             --tasks ${tasks}
#             # gsm8k
#     else
#         echo "Checkpoint does not exist: $CKPT_PATH, skipping."
#     fi
# done





# for i in "${checkpoints[@]}"; do
#     CKPT_PATH="${STEP1_PTH_PATH}/ckpt-${i}.pth"
#     if [ -f "$CKPT_PATH" ]; then
#         echo "Evaluating checkpoint: $CKPT_PATH"
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         python run_lm_eval.py \
#             -c ${qwen_yaml} \
#             -c ${la_yaml} \
#             --path "${CKPT_PATH}" \
#             --is_pretrained no \
#             --bsz ${bsz} \
#             --tokenizer_name ${tokenizer} \
#             --tasks ${tasks}
#             # gsm8k
#     else
#         echo "Checkpoint does not exist: $CKPT_PATH, skipping."
#     fi
# done






# bsz=1
# for i in "${checkpoints[@]}"; do
#     # CKPT_PATH="${STEP0_PTH_PATH}/ckpt-${i}.pth"
#     CKPT_PATH="/mnt/nfs/aman/models/20260104__qwen3-8b_gdn_convF_rmsT_l2T_negEigT-4-2k/ckpt-${i}.pth"
#     if [ -f "$CKPT_PATH" ]; then
#         echo "Evaluating checkpoint: $CKPT_PATH"
#         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
#         python run_lm_eval.py \
#             -c ${qwen_yaml} \
#             -c ${la_yaml} \
#             --path "${CKPT_PATH}" \
#             --is_pretrained no \
#             --bsz ${bsz} \
#             --tokenizer_name ${tokenizer} \
#             --tasks ${tasks} \
#             --log_path logs/20251231/bsz1
#             # gsm8k
#     else
#         echo "Checkpoint does not exist: $CKPT_PATH, skipping."
#     fi
# done

# python ~/exp.py --gpus 0



echo "Testing evaluation on: $TEST_CKPT_PATH"



for i in "${checkpoints[@]}"; do
    CKPT_PATH="/mnt/nfs/aman/models/20260104__qwen3-8b_gdn_convF_rmsT_l2T_negEigT-4-2k/ckpt-${i}.pth"
    if [ -f "$CKPT_PATH" ]; then
        echo "Checkpoint exists. Running evaluation..."
        echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        # Count number of GPUs from CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
        echo "Using ${NUM_GPUS} GPUs with DistributedDataParallel via torchrun"
        torchrun --nproc-per-node=${NUM_GPUS} multi_gpu_run_lm_eval.py \
            -c ${qwen_yaml} \
            -c ${la_yaml} \
            --path "${CKPT_PATH}" \
            --is_pretrained no \
            --bsz 4 \
            --precision bf16 \
            --tokenizer_name ${tokenizer} \
            --tasks ${tasks}

    else
        echo "Checkpoint does not exist: $TEST_CKPT_PATH"
        exit 1
    fi
done