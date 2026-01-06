# Changes Summary and Setup Status

## Overview
This document summarizes all changes made to set up the GDN (Gated Delta Net) training experiment with 2048 context length and multiple datasets for the RADLADS 3-step distillation process.

## Files Created

### Scripts Directory (`scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/`)
1. **vars.sh** - Environment variables and configuration
   - CUDA device assignments
   - Paths for models, outputs, and datasets
   - WandB configuration (project: `hybrid_training`, API key)
   - Run name and step directory names

2. **step0.sh** - RADLADS Step 0 (Initial Distillation)
   - Loads pretrained Qwen3-8B-Base model
   - Initializes GDN student model
   - Uses `distill1.yaml` config

3. **step1.sh** - RADLADS Step 1 (With Teacher Model)
   - Loads checkpoint from Step 0
   - Uses teacher model for knowledge distillation
   - Uses `distill2.yaml` config

4. **step2.sh** - RADLADS Step 2 (Final Fine-tuning)
   - Loads checkpoint from Step 1
   - Final fine-tuning stage
   - Uses `distill3.yaml` config

5. **all_steps.sh** - Run all steps sequentially

6. **README.md** - General documentation and setup instructions

7. **SETUP_GUIDE.md** - Detailed setup guide with explanations

8. **QUICK_START.md** - Quick reference for running experiments

9. **CHANGES_SUMMARY.md** - This file

### Configs Directory (`configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/`)
1. **gdn.yaml** - GDN-specific model configuration
   - `tmix: qwen3gdn`
   - `attention_type: gdn`
   - `mode: chunk`
   - `allow_neg_eigval: true`
   - `use_short_conv: false`
   - `use_qk_rmsnorm: true`
   - `use_qk_l2norm_in_kernel: true`

2. **distill1.yaml** - Step 0 training configuration
   - `ctx_len: 2048`
   - Dataset: `stag0_100M_binidx`
   - `my_exit_tokens: 99979300`
   - `magic_prime: 48809`
   - `wandb: hybrid_training`
   - `micro_bsz: 2` (reduced for 2048 ctx_len)
   - `attention_distillation_stage: 1`

3. **distill2.yaml** - Step 1 training configuration
   - `ctx_len: 2048`
   - Dataset: `stag1_500M_binidx`
   - `my_exit_tokens: 499844135`
   - `magic_prime: 244043`
   - `wandb: hybrid_training`
   - `micro_bsz: 4`
   - `attention_distillation_stage: 2`
   - Teacher configuration with KL loss

4. **distill3.yaml** - Step 2 training configuration
   - `ctx_len: 2048`
   - Dataset: `stag1_500M_binidx` (same as step 1)
   - `my_exit_tokens: 499844135`
   - `magic_prime: 244043`
   - `wandb: hybrid_training`
   - `micro_bsz: 4`
   - `strategy: fsdp`

5. **qwen3-8b-instructteacher.yaml** - Teacher model configuration
   - Path: `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth`
   - `ctx_len: 2048`
   - Qwen3 model architecture settings

### Root Directory
1. **convert_arrow_to_binidx.py** - Dataset conversion script
   - Converts Arrow format (HuggingFace datasets) to binidx format
   - Handles pre-tokenized datasets (detects `input_ids` column)
   - Calculates magic_prime automatically
   - Supports custom tokenizers and context lengths

## Dataset Conversions Performed

### stag0_100M → binidx
- **Input**: Arrow format, 48,829 examples
- **Output**: `stag0_100M_binidx.bin` (382MB) + `.idx` (954KB)
- **Total tokens**: 99,979,300
- **Magic prime**: 48,809 (for ctx_len 2048)
- **Used in**: Step 0 (distill1.yaml)

### stag1_500M → binidx
- **Input**: Arrow format, 244,141 examples
- **Output**: `stag1_500M_binidx.bin` (1.9GB) + `.idx` (4.7MB)
- **Total tokens**: 499,844,135
- **Magic prime**: 244,043 (for ctx_len 2048)
- **Used in**: Step 1 (distill2.yaml) and Step 2 (distill3.yaml)

## Key Configuration Changes

### Context Length
- All configs updated from `ctx_len: 512` to `ctx_len: 2048`
- Batch sizes reduced accordingly:
  - Step 0: `micro_bsz: 2` (was 4)
  - Step 1: `micro_bsz: 4` (was 12)
  - Step 2: `micro_bsz: 4` (was 12)

### Dataset Paths
- Step 0: `/mnt/nfs/parsa/data/radlad_data_steps/stag0_100M_binidx`
- Step 1: `/mnt/nfs/parsa/data/radlad_data_steps/stag1_500M_binidx`
- Step 2: `/mnt/nfs/parsa/data/radlad_data_steps/stag1_500M_binidx`

### WandB Configuration
- Project: `hybrid_training` (updated from `qwen2rwkv`)
- API key: Configured in `vars.sh`
- Environment variables exported automatically

### Output Paths
- Updated to use `/work/parsa/RADLADS-paper/out/pths/` instead of `/work/parsa/radlads/out/pths/`

## Environment Setup

### New Conda Environment: radlads2
- **Created**: New conda environment `radlads2` with Python 3.11
- **Purpose**: Resolve Triton/FLA compilation errors from the original `radlads` environment
- **Installation**: Used `scripts/pip_installs.sh` to install all packages
- **Package Versions**:
  - PyTorch: 2.6.0+cu126
  - Triton: 3.2.0
  - FLA (flash-linear-attention): 0.4.2 (installed from git)
  - Lightning: 2.6.0
  - DeepSpeed: 0.18.3
  - Transformers: 4.53.3
  - Datasets: 3.6.0
- **Status**: ✅ Verified - All packages installed and imports working correctly

### Datasets Library
- Upgraded from `datasets 3.6.0` to `datasets 4.4.1` to fix Arrow format loading issues

### Conda Environment (Legacy)
- **Note**: The original `radlads` environment had Triton compilation errors
- **Solution**: Use `radlads2` environment instead
- All scripts should now use `radlads2` conda environment

## What's Left To Do

### 1. Convert Pretrained Model (REQUIRED)
The HuggingFace model needs to be converted to `.pth` format:

```bash
mkdir -p /work/parsa/pths/Qwen3-8B-Base
conda activate radlads2
python3 convert_hf_to_pth.py \
    /work/parsa/data/models/Qwen3-8B-Base \
    /work/parsa/pths/Qwen3-8B-Base/pretrained.pth
```

**Status**: ⚠️ **NOT DONE** - Required before running step0.sh

**Time estimate**: 5-10 minutes

### 2. Verify Model Path
Ensure the converted model exists at:
- `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth`

**Status**: ⚠️ **PENDING** - Depends on step 1

### 3. Run Step 0
Once the model is converted:

```bash
cd /work/parsa/RADLADS-paper
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step0.sh
```

**Status**: ⚠️ **READY TO RUN** - Waiting for model conversion

### 4. Run Step 1 (After Step 0 Completes)
```bash
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step1.sh
```

**Status**: ⚠️ **PENDING** - Depends on step 0 completion

### 5. Run Step 2 (After Step 1 Completes)
```bash
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step2.sh
```

**Status**: ⚠️ **PENDING** - Depends on step 1 completion

## Configuration Verification Checklist

- [x] Scripts directory created with all step scripts
- [x] Configs directory created with all YAML files
- [x] Dataset stag0_100M converted to binidx
- [x] Dataset stag1_500M converted to binidx
- [x] Magic primes calculated for both datasets
- [x] Config files updated with correct dataset paths
- [x] Config files updated with correct token counts
- [x] Config files updated with correct magic primes
- [x] WandB project and API key configured
- [x] Context length set to 2048 in all configs
- [x] Batch sizes adjusted for 2048 context length
- [x] Output paths configured correctly
- [ ] **Pretrained model converted to .pth format** ⚠️
- [ ] **Model path verified** ⚠️
- [ ] **Step 0 executed successfully** ⚠️
- [ ] **Step 1 executed successfully** ⚠️
- [ ] **Step 2 executed successfully** ⚠️

## Important Notes

### Model Paths
- **Step 0**: Uses `--train.load_model` pointing to `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth`
- **Steps 1-2**: Use teacher model from `qwen3-8b-instructteacher.yaml` (same path)

### Dataset Usage
- **Step 0**: Uses `stag0_100M` (100M tokens) - smaller dataset for initial distillation
- **Step 1**: Uses `stag1_500M` (500M tokens) - larger dataset for refinement
- **Step 2**: Uses `stag1_500M` (500M tokens) - same as step 1 for final tuning

### Memory Considerations
- Batch sizes are reduced for 2048 context length (4x larger than default 512)
- If you encounter OOM errors, reduce `micro_bsz` further in the config files
- Consider using gradient checkpointing (`grad_cp: 1`) if needed

### WandB Logging
- All training runs will log to the `hybrid_training` project
- Metrics will be tracked automatically
- Check WandB dashboard for training progress

## Next Steps

1. **Convert the model** (if not already done)
2. **Run step0.sh** to start the training pipeline
3. **Monitor training** via WandB or console output
4. **Run step1.sh** after step 0 completes
5. **Run step2.sh** after step 1 completes

## Troubleshooting

### Triton/FLA Compilation Error (RESOLVED)
- **Error**: `'arith.mulf' op requires the same encoding for all operands and results` in FLA library
- **Cause**: Package version incompatibility in the original `radlads` environment
- **Solution**: Created new `radlads2` environment with properly matched package versions
  - Python 3.11 (recommended by FLA)
  - PyTorch 2.6.0+cu126
  - Triton 3.2.0
  - FLA 0.4.2 (from git)
- **Status**: ✅ Fixed - Use `radlads2` environment for all training scripts

### Other Issues
If you encounter issues:
- Check that the model is converted and at the correct path
- Verify dataset files exist (.bin and .idx)
- Check CUDA device availability
- Review batch sizes if OOM errors occur
- Check WandB API key is valid
- **Important**: Always use `conda activate radlads2` before running training scripts

## Files Reference

- **Conversion script**: `/work/parsa/RADLADS-paper/convert_arrow_to_binidx.py`
- **Main training script**: `/work/parsa/RADLADS-paper/train.py`
- **Model conversion**: `/work/parsa/RADLADS-paper/convert_hf_to_pth.py`

---

**Last Updated**: 2025-01-05
**Status**: Ready for model conversion and training

