from modelscope import snapshot_download

model_dir = snapshot_download(
    'LULALULALU/VOSR_CKPT',
    local_dir='preset/ckpts'
)

print(model_dir)
