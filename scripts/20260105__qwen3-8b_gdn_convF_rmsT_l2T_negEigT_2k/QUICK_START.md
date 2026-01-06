# Quick Start: Running Step 0

## ⚠️ Important: Your Dataset Format

Your dataset at `/mnt/nfs/parsa/data/radlad_data_steps/stag0_100M/` is in **Arrow format** (HuggingFace datasets), but the training code expects **binidx format** (`.bin` and `.idx` files).

You have two options:

### Option 1: Convert Arrow to binidx (Recommended)
You'll need to convert your Arrow dataset to binidx format. Check if there's a conversion script or use `make_data_hf.py` to create the binidx format.

### Option 2: Check if binidx already exists elsewhere
The dataset might already be converted somewhere else. Look for files ending in `.bin` and `.idx`.

## Required Changes Before Running

### 1. Convert Model Format ✅

Your model is at `/work/parsa/data/models/Qwen3-8B-Base/` (HuggingFace format). Convert it:

```bash
mkdir -p /work/parsa/pths/Qwen3-8B-Base
python3 convert_hf_to_pth.py \
    /work/parsa/data/models/Qwen3-8B-Base \
    /work/parsa/pths/Qwen3-8B-Base/pretrained.pth
```

**Time**: ~5-10 minutes

### 2. Convert Dataset Format ⚠️

Your dataset needs to be in binidx format. If you have the Arrow dataset, you may need to:
- Use `make_data_hf.py` to convert it
- Or find where the binidx version is stored

Once you have the binidx dataset, update the path in:
- `configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/distill1.yaml`
- `configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/distill2.yaml`
- `configs/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/distill3.yaml`

### 3. Calculate Magic Prime

Once you have the binidx dataset:

```bash
python3 make_data_hf.py check <path_to_binidx_dataset> 2048
```

Update the `magic_prime` and `my_exit_tokens` values in all three `distill*.yaml` files.

### 4. Verify Teacher Model Path

The teacher model path in `qwen3-8b-instructteacher.yaml` is currently:
```yaml
path: /work/parsa/pths/Qwen3-8B-Base/pretrained.pth
```

This should match where you saved the converted model (from step 1).

## Understanding the Two Model Paths

### Path 1: Initial Model (Step 0)
- **Where**: `step0.sh` line 11: `--train.load_model ${pths_dir}/Qwen3-8B-Base/pretrained.pth`
- **Purpose**: Initializes the GDN student model
- **What it does**: Loads pretrained weights into your GDN model (partial load - attention layers get replaced)

### Path 2: Teacher Model (Steps 1-2)
- **Where**: `qwen3-8b-instructteacher.yaml` line 3: `path: /work/parsa/pths/Qwen3-8B-Base/pretrained.pth`
- **Purpose**: Provides supervision for knowledge distillation
- **What it does**: The original model "teaches" your GDN model how to behave

**For Step 0**: You only need Path 1 (the initial model). Path 2 is only used in steps 1-2.

## Current Configuration Status

✅ **Model conversion needed**: Yes - convert HF model to .pth  
⚠️ **Dataset format**: Arrow → needs binidx conversion  
⚠️ **Magic prime**: Needs calculation after dataset conversion  
✅ **Paths configured**: Mostly correct, just need to verify after conversion  

## Next Steps

1. **Convert the model** (5-10 min)
2. **Convert/find the binidx dataset** (time varies)
3. **Calculate magic_prime** (1 min)
4. **Update config files** with magic_prime values (1 min)
5. **Run step0.sh** 🚀

## Running Step 0

Once everything is ready:

```bash
cd /work/parsa/RADLADS-paper
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step0.sh
```

## What to Expect

- **Training time**: Depends on dataset size and hardware
- **Checkpoints**: Saved to `/work/parsa/RADLADS-paper/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-1/`
- **Output files**: `rwkv-<epoch>.pth` and `rwkv-final.pth`

## Troubleshooting

**"File not found: pretrained.pth"**
→ Run the model conversion (step 1 above)

**"Dataset not found" or "Invalid dataset format"**
→ Convert Arrow dataset to binidx format

**"magic_prime validation failed"**
→ Recalculate using `make_data_hf.py check`

**"Out of memory"**
→ Reduce `micro_bsz` in `distill1.yaml` (try `micro_bsz: 1`)

