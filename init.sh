#!/bin/bash

#VARIABLES
TOKEN="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"

# This script is used to initialize the environment for the
downloadRequirements() {
    apt update && apt install -y git git-lfs unzip nano wget p7zip-full ninja-build
    git clone https://github.com/TheLastBen/diffusers
    pip install --no-cache-dir ./diffusers triton==2.0.0.dev20220701
    pip install -U -r ./diffusers/examples/dreambooth/requirements.txt
    pip install -q OmegaConf accelerate==0.12.0 bitsandbytes scipy
    #pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/T4/xformers-0.0.13.dev0-py3-none-any.whl
    git clone https://github.com/facebookresearch/xformers/
    cd xformers
    git submodule update --init --recursive
    #set MAX_JOBS as threads for multithread ninja-build
    FORCE_CUDA=1 MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" pip install --verbose --no-deps -e .
    cd /workspace
}

downloadModel() {
    mkdir ./stable-diffusion-v1-5
    cd ./stable-diffusion-v1-5
    git init
    git lfs install --system --skip-repo
    git remote add -f origin "https://USER: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@huggingface.co/runwayml/stable-diffusion-v1-5"
    git config core.sparsecheckout true
    echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json" >.git/info/sparse-checkout
    git pull origin main
    git clone "https://USER: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@huggingface.co/stabilityai/sd-vae-ft-mse" ./vae
    rm -rf ./.git
    cd ../
}

train() {
    Seed=96576
    Training_Steps=3000
    Train_text_encoder_for=10
    precision="fp16"
    MODELT_NAME="./stable-diffusion-v1-5/"
    INSTANCE_DIR="/volume/photos"
    OUTPUT_DIR="/volume/output"
    PT=""

    stptxt=$(($Training_Steps * $Train_text_encoder_for / 100))

    accelerate launch ./train_dreambooth.py \
        --train_text_encoder \
        --pretrained_model_name_or_path="$MODELT_NAME" \
        --instance_data_dir="$INSTANCE_DIR" \
        --output_dir="$OUTPUT_DIR" \
        --instance_prompt="$PT" \
        --seed=$Seed \
        --resolution=512 \
        --mixed_precision=$precision \
        --train_batch_size=1 \
        --gradient_accumulation_steps=1 \
        --use_8bit_adam \
        --learning_rate=2e-6 \
        --lr_scheduler="polynomial" \
        --center_crop \
        --lr_warmup_steps=0 \
        --max_train_steps=$Training_Steps
}
downloadRequirements
#downloadModel
#train
