source "$(dirname "$0")/vars.sh"
echo $RUN_NAME

# Create logs directory if it doesn't exist
SCRIPT_DIR="$(dirname "$0")"
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}/step0_${TIMESTAMP}.log"

echo "Logging output to: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
RWKV_TORCH_COMPILE=0 \
RWKV_JIT_ON=0 \
python3 train.py \
    -c ${qwen_yaml} \
    -c ${la_yaml} \
    -c configs/${RUN_NAME}/distill1.yaml \
    --train.load_model ${pths_dir}/Qwen3-8B-Base/pretrained.pth \
    --train.proj_name ${RUN_NAME} \
    2>&1 | tee "${LOG_FILE}"

