export USERNAME=aman
model_name="20260105__qwen3-8b_gdn_convF_rmsT_l2T_negEigT_2k"
pth=/work/${USERNAME}/radlads/pths
out_pths_dir=/work/${USERNAME}/radlads/out/pths
out_hf_dir=/work/${USERNAME}/radlads/out/hf

# python convert_to_safetensors.py \
#     ${out_pths_dir}/${model_name}-1/rwkv-final.pth \
#     ${out_hf_dir}/${model_name}-1.safetensors

# python convert_to_safetensors.py \
#     ${out_pths_dir}/${model_name}-2/rwkv-final.pth \
#     ${out_hf_dir}/${model_name}-2.safetensors

python convert_to_safetensors.py \
    ${out_pths_dir}/${model_name}-1/ckpt-final.pth \
    ${out_hf_dir}/${model_name}-1_debug.safetensors
