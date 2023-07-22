FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

RUN apt update
RUN apt install git -y 
RUN apt install wget -y 
RUN apt install python3 python3-pip -y



# Install dependencies (one-by-one for better caching)
#RUN pip install --upgrade pip
RUN pip install torch
RUN pip install transformers
RUN pip install datasets
RUN pip install evaluate
RUN pip install xformers
RUN pip install wandb
RUN pip install peft 
RUN pip install trl 
RUN pip install scipy 
RUN pip install accelerate 
RUN pip install scikit-learn
RUN pip install pandas 
RUN pip install bleurt@https://github.com/google-research/bleurt/archive/b610120347ef22b494b6d69b4316e303f5932516.zip#egg=bleurt
RUN pip install matplotlib
RUN pip install bert_score

RUN git clone https://github.com/EleutherAI/lm-evaluation-harness
RUN pip install -e lm-evaluation-harness

RUN git clone https://github.com/timdettmers/bitsandbytes.git
# CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
# make argument in {cuda110, cuda11x, cuda12x}
# if you do not know what CUDA you have, try looking at the output of: python -m bitsandbytes
ENV CUDA_VERSION=117
#RUN cd bitsandbytes && git checkout b844e104b79ddc06161ff975aa93ffa9a7ec4801
RUN cd bitsandbytes && make cuda11x
RUN cd bitsandbytes && python3 setup.py install
#RUN pip install bitsandbytes
#RUN python3 check_bnb_install.py

ENV HF_DATASETS_CACHE="/hf_cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/hf_cache/hub"

ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

# Copy the code
COPY . /code

# Install a useful helper to check bitsandbytes installation. Only works at runtime.
RUN wget https://gist.githubusercontent.com/TimDettmers/1f5188c6ee6ed69d211b7fe4e381e713/raw/4d17c3d09ccdb57e9ab7eca0171f2ace6e4d2858/check_bnb_install.py 

# Set the working directory
WORKDIR /code