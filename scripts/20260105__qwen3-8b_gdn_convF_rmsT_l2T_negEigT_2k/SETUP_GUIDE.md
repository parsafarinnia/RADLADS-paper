# Setup Guide: Running Your First GDN Training Experiment

## Understanding the Two Model Paths

There are **two different model paths** used in the RADLADS distillation process:

### 1. **Initial Pretrained Model** (for Step 0)
- **Location**: `step0.sh` uses `--train.load_model ${pths_dir}/Qwen3-8B-Base/pretrained.pth`
- **Purpose**: This is the **starting point** for your student model. Step 0 initializes the GDN student model with weights from this pretrained Qwen3-8B-Base model.
- **Current path**: `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth`
- **Your model location**: `/work/parsa/data/models/Qwen3-8B-Base/` (HuggingFace format)

### 2. **Teacher Model** (for Steps 1-2)
- **Location**: `qwen3-8b-instructteacher.yaml` has `train.teacher.path`
- **Purpose**: This is the **teacher model** used for knowledge distillation in steps 1-2. It provides supervision signals to train the student.
- **Current path**: `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth`
- **Note**: For steps 1-2, you can use the same pretrained model or a fine-tuned instruct version

## Why `.pth` Format?

The training code expects PyTorch `.pth` files (state dictionaries), not HuggingFace safetensors. You need to convert your model.

## Step-by-Step Setup

### Step 1: Convert HuggingFace Model to .pth Format

Your model is at `/work/parsa/data/models/Qwen3-8B-Base/` in HuggingFace format. Convert it:

```bash
# Create the directory
mkdir -p /work/parsa/pths/Qwen3-8B-Base

# Convert the model
python3 convert_hf_to_pth.py \
    /work/parsa/data/models/Qwen3-8B-Base \
    /work/parsa/pths/Qwen3-8B-Base/pretrained.pth
```

**What this does**: 
- Loads the HuggingFace model
- Extracts the state dictionary (all model weights)
- Saves it as a `.pth` file that the training code can load

**Time estimate**: ~5-10 minutes for an 8B model

### Step 2: Calculate Magic Prime for Your Dataset

The `magic_prime` is a special prime number used for pseudo-random dataset sampling. You **must** calculate this before training:

```bash
python3 make_data_hf.py check /mnt/nfs/parsa/data/radlad_data_steps/stag0_100M 2048
```

**What this outputs**:
- The `magic_prime` value
- The `my_exit_tokens` value (total tokens in your dataset)

**Example output**:
```
### magic_prime = 48827 (for ctxlen 2048)
--my_exit_tokens 100000000 --magic_prime 48827 --ctx_len 2048
```

### Step 3: Update Configuration Files

After getting the magic_prime, update these files:

#### Update `distill1.yaml`:
```yaml
my_exit_tokens: <value from make_data_hf.py>
magic_prime: <value from make_data_hf.py>
```

#### Update `distill2.yaml` and `distill3.yaml`:
Same values (or adjust `my_exit_tokens` if you want to use different amounts per step)

### Step 4: Verify Paths

Check that these paths are correct:

1. **Pretrained model** (for step0.sh):
   - Should exist: `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth`
   - Set in: `step0.sh` line 11

2. **Teacher model** (for steps 1-2):
   - Should exist: `/work/parsa/pths/Qwen3-8B-Base/pretrained.pth` (or your instruct model)
   - Set in: `qwen3-8b-instructteacher.yaml` line 3

3. **Dataset**:
   - Should exist: `/mnt/nfs/parsa/data/radlad_data_steps/stag0_100M.bin` and `.idx`
   - Set in: `distill1.yaml`, `distill2.yaml`, `distill3.yaml` line 12

4. **Output directory**:
   - Will be created: `/work/parsa/RADLADS-paper/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-1/`
   - Set in: `vars.sh` line 5

### Step 5: Adjust Batch Sizes (Optional)

The current batch sizes are conservative for 2048 context length:
- Step 0: `micro_bsz: 2` (total batch size = 2 × 8 GPUs = 16)
- Step 1: `micro_bsz: 4` (total batch size = 4 × 8 GPUs = 32)
- Step 2: `micro_bsz: 4` (total batch size = 4 × 8 GPUs = 32)

If you have more GPU memory, you can increase these in the `distill*.yaml` files.

## Running Step 0

Once everything is set up:

```bash
cd /work/parsa/RADLADS-paper
bash scripts/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k/step0.sh
```

## What Happens in Step 0?

1. **Loads pretrained model**: Initializes student GDN model with Qwen3-8B-Base weights
2. **Partial loading**: `load_partial: 1` means it only loads compatible weights (attention layers get replaced with GDN)
3. **Training**: Distills knowledge from the pretrained model to the GDN student
4. **Saves checkpoints**: To `/work/parsa/RADLADS-paper/out/pths/20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k-1/`

## Monitoring Training

- **Checkpoints**: Saved as `rwkv-<epoch>.pth` and `rwkv-final.pth`
- **Logs**: Check console output for loss values
- **Wandb**: If configured, training metrics will be logged (wandb project: `qwen2rwkv`)

## Common Issues and Solutions

### Issue: "File not found: pretrained.pth"
**Solution**: Run the conversion script (Step 1 above)

### Issue: "magic_prime validation failed"
**Solution**: Recalculate magic_prime using `make_data_hf.py check`

### Issue: "Out of memory"
**Solution**: Reduce `micro_bsz` in the distill config files

### Issue: "Dataset not found"
**Solution**: Verify the dataset path and that both `.bin` and `.idx` files exist

## Learning Tips

1. **Understand the three stages**:
   - **Step 0**: Initialize GDN student from pretrained model
   - **Step 1**: Distill with teacher supervision (attention distillation)
   - **Step 2**: Final fine-tuning

2. **Magic Prime**: This is a clever trick from LinearAttentionArena for efficient dataset sampling. It ensures good coverage without storing full indices.

3. **Partial Loading**: `load_partial: 1` in step 0 means only compatible layers are loaded. The attention mechanism changes from softmax to GDN, so those weights are reinitialized.

4. **Context Length**: 2048 is 4x larger than the default 512, so:
   - Memory usage is ~4x higher
   - Batch sizes should be reduced
   - Training will be slower

5. **Teacher vs Student**: 
   - Teacher = original pretrained model (softmax attention)
   - Student = your GDN model (linear attention)
   - Distillation transfers knowledge from teacher to student

## Next Steps After Step 0

Once step 0 completes successfully:
1. Check the final checkpoint: `ckpt-final.pth`
2. Run step 1: `bash step1.sh` (uses teacher model for distillation)
3. Run step 2: `bash step2.sh` (final fine-tuning)

Good luck with your training! 🚀

