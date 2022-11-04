FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
WORKDIR /
ENV FORCE_CUDA=1 MAX_JOBS=3 TORCH_CUDA_ARCH_LIST="5.0;5.2;6.0;6.1+PTX;7.0;7.5+PTX;8.0;8.6+PTX" python_abi=cp37-cp37m
RUN apt update && apt install -y git git-lfs unzip nano wget p7zip-full
RUN pip install bitsandbytes
RUN pip install scipy
#install precompiled xformers for your GPU or compile from source https://github.com/facebookresearch/xformers/issues/473#issuecomment-1272576184
RUN pip install https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl
#RUN pip install xformers

# Fix waifu diffusion training.

WORKDIR /train
RUN git clone https://github.com/TheLastBen/diffusers && \
    pip install --no-cache-dir ./diffusers triton==2.0.0.dev20220701 &&\
    pip install -U -r ./diffusers/examples/dreambooth/requirements.txt && \
    cd diffusers/examples/dreambooth/ && \
    cp train_dreambooth.py /train/ && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /train/diffusers
COPY stable-diffusion-v1-5.7z /train/
RUN 7z x stable-diffusion-v1-5.7z && rm stable-diffusion-v1-5.7z

COPY photos.zip /train/
RUN unzip photos.zip -d photos
COPY init.sh /train/
ENV HF_HOME=/train/.hub