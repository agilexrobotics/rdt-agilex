python -m scripts.inference-server \
    --pretrained_model_name_or_path="/home/agilex/checkpoint_rdt/rdt-finetune-1b/checkpoint-8000" \
    --pretrained_vision_encoder_name_or_path="/home/agilex/models/google/siglip-so400m-patch14-384" \
    --ctrl_freq=25    # your control frequency
