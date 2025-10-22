# RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation

Referencing https://github.com/thu-ml/RoboticsDiffusionTransformer, you can train the model using this repository and then use the inference method provided in the repository. They are completely consistent in terms of the model. For more details, please refer to this link.

## Installation
ubuntu22.04 + cuda12.8
1. Clone this repo and install prerequisites:

    ```bash
    # Clone this repo
    git clone https://github.com/agilexrobotics/rdt-agilex.git RoboticsDiffusionTransformer
    git clone https://github.com/NVIDIA/cutlass.git
    cd RoboticsDiffusionTransformer
    
    # Create a Conda environment
    conda create -n rdt python=3.10.0
    conda activate rdt
    
    # Install pytorch
    # Look up https://pytorch.org/get-started/previous-versions/ with your cuda version for a correct command
    pip install torch==2.1.0 torchvision==0.16.0  --index-url https://download.pytorch.org/whl/cu121
    
    # Install packaging
    pip install packaging==24.0
    
    # Install flash-attn
    pip install flash-attn --no-build-isolation
    
    # Install other prequisites
    pip install -r requirements.txt
    ```

2. Download off-the-shelf multi-modal encoders:

   You can download the encoders from the following links:

   - `t5-v1_1-xxl`: [link](https://huggingface.co/google/t5-v1_1-xxl/tree/main)🤗
   - `siglip`: [link](https://huggingface.co/google/siglip-so400m-patch14-384)🤗
   - `rdt-1b`: [link](https://huggingface.co/robotics-diffusion-transformer/rdt-1b)🤗
## Fine-Tuning on Your Own Dataset


1. Prepare your agilex dataset

2. Implement the dataset loader:
   ```bash
   gedit data/hdf5_vla_dataset.py
   ```
   Modify the HDF5_DIRS parameter to your dataset path.

3. Compute the dataset statistics information for dataset:
   ```bash
   # Under the root directory of this repo
   # Use -h to see the full usage
   python -m data.compute_dataset_stat_hdf5
   ```

4. Start fine-tuning:

   ```bash
   gedit finetune.sh
   ```
   TEXT_ENCODER_NAME: t5-v1_1-xxl path

   VISION_ENCODER_NAME: siglip path

   CUTLASS_PATH: cutlass path

   pretrained_model_name_or_path: rdt-1b path

   OUTPUT_DIR: model output path

   train_batch_size: train batch size

   sample_batch_size: sample batch size

   Use this to start fine-tuning:

   ```bash
   source finetune.sh
   ```

5. inference:
   ```bash
   gedit finetune.sh
   ```
   pretrained_model_name_or_path: your finetuning model path

   pretrained_vision_encoder_name_or_path: siglip path

   Use this to start fine-tuning:

   ```bash
   bash inference.sh
   ```
## License

All the code, model weights, and data are licensed under [MIT license](./LICENSE).
