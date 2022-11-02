# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel AS builder
WORKDIR /
ENV FORCE_CUDA=1 MAX_JOBS=3 TORCH_CUDA_ARCH_LIST="5.0;5.2;6.0;6.1+PTX;7.0;7.5+PTX;8.0;8.6+PTX" python_abi=cp37-cp37m
RUN apt-get update && apt-get install -y git git-lfs nano wget unzip
RUN pip3 wheel bitsandbytes 

FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
WORKDIR /
ADD https://api.github.com/repos/ShivamShrirao/diffusers/git/refs/heads/main /version.json
RUN --mount=type=bind,target=whls,from=builder apt-get update && apt-get install -y git git-lfs unzip nano wget
#install precompiled xformers for your GPU or compile from source https://github.com/facebookresearch/xformers/issues/473#issuecomment-1272576184
RUN git clone https://github.com/ShivamShrirao/diffusers && \
    cd diffusers/examples/dreambooth/ && \
    pip install --no-cache-dir /diffusers triton==2.0.0.dev20220701 /whls/bitsandbytes*.whl && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl && \
    cp train_dreambooth.py /train/ && \
    rm -rf /var/lib/apt/lists/*
# Fix waifu diffusion training.
RUN pip install scipy
WORKDIR /train
RUN mkdir stable-diffusion-v1-5 && cd stable-diffusion-v1-5 && \
    git init && git lfs install --system --skip-repo && \
    git remote add -f origin "https://USER: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@huggingface.co/runwayml/stable-diffusion-v1-5" && \
    git config core.sparsecheckout true && \
    echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json" >.git/info/sparse-checkout && \
    git pull origin main && \
    git clone "https://USER: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@huggingface.co/stabilityai/sd-vae-ft-mse" ./vae

COPY photos.zip /train/
RUN unzip photos.zip -d photos
COPY init.sh /train/
ENV HF_HOME=/train/.hub