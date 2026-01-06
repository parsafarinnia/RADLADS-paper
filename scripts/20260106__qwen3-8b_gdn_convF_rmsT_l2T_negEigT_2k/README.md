# GDN Training Scripts for 2048 Context Length

This directory contains training scripts for GDN (Gated Delta Net) models using the RADLADS 3-step distillation process with 2048 context length.

## Setup Instructions

### 1. Calculate Magic Prime

Before running training, you need to calculate the `magic_prime` value for your dataset. Run:

```bash
python3 make_data_hf.py check /mnt/nfs/parsa/data/radlad_data_steps/stag0_100M 2048
```

This will output the required `magic_prime` value and `my_exit_tokens`. Update these values in:
- `configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/distill1.yaml`
- `configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/distill2.yaml`
- `configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/distill3.yaml`

### 2. Verify Pretrained Model Path

Ensure the pretrained Qwen3-8B-Base model is available at:
```
/work/parsa/pths/Qwen3-8B-Base/pretrained.pth
```

If not, convert from HuggingFace format:
```bash
python3 convert_hf_to_pth.py <HF_MODEL_PATH> /work/parsa/pths/Qwen3-8B-Base/pretrained.pth
```

**Note**: The teacher model path is configured in `qwen3-8b-instructteacher.yaml`. If your path is different, you can override it via command line:
```bash
--train.teacher.path /path/to/your/pretrained.pth
```

### 3. Update Dataset Paths (if using multiple datasets)

If you want to use different datasets for different steps, update the `data_file` path in:
- `distill1.yaml` (Step 0)
- `distill2.yaml` (Step 1)
- `distill3.yaml` (Step 2)

### 4. Adjust Batch Sizes (if needed)

The current batch sizes are set conservatively for 2048 context length:
- Step 0: `micro_bsz: 2`
- Step 1: `micro_bsz: 4`
- Step 2: `micro_bsz: 4`

Adjust these in the distill configs if you have more GPU memory available.

## Running Training

### Run All Steps Sequentially

```bash
cd /work/parsa/RADLADS-paper
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/all_steps.sh
```

### Run Individual Steps

```bash
# Step 0: Initial distillation
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step0.sh

# Step 1: With teacher model
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step1.sh

# Step 2: Final fine-tuning
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step2.sh
```

## Output Locations

Checkpoints will be saved to:
- Step 0: `/work/parsa/radlads/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-1/`
- Step 1: `/work/parsa/radlads/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-2/`
- Step 2: `/work/parsa/radlads/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-3/`

## Magic Prime Calculation Details

The `magic_prime` is calculated as:
- Find the largest prime number of the form `3n+2` that is less than `(dataset_size / ctx_len) - 1`
- For 2048 context length: `magic_prime < (dataset_size / 2048) - 1`

The `make_data_hf.py check` command automatically finds this value for you.

## Multiple Datasets

To use multiple datasets, you have three options:

1. **Separate runs**: Create different script directories for each dataset
2. **Sequential steps**: Use different datasets in `distill1.yaml`, `distill2.yaml`, `distill3.yaml`
3. **Combined dataset**: Merge datasets into one binidx file before training

## Notes

- GDN uses chunk mode for training (configured in `gdn.yaml`)
- The model uses `allow_neg_eigval: true` for better performance
- Short convolution is disabled (`use_short_conv: false`)
- QK RMSNorm and L2 normalization in kernel are enabled

