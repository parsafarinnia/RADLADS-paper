source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

# Create logs directory if it doesn't exist
SCRIPT_DIR="$(dirname "$0")"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/step1_${TIMESTAMP}.log"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c ${qwen_yaml} \
    -c ${la_yaml} \
    -c configs/${RUN_NAME}/qwen3-8b-instructteacher.yaml \
    -c configs/${RUN_NAME}/distill2.yaml \
    --train.load_model ${out_pths_dir}/${RUN_NAME}-1/ckpt-final.pth \
    --train.proj_name ${RUN_NAME} \
    > "${LOG_FILE}" 2>&1

