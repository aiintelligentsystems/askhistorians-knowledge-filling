FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Install dependencies (one-by-one for better caching)
RUN pip install --upgrade pip
RUN pip install transformers
RUN pip install datasets
RUN pip install evaluate
RUN pip install xformers
RUN pip install wandb
RUN pip install peft 
RUN pip install trl 
RUN pip install bitsandbytes
RUN pip install scipy 
RUN pip install accelerate 
RUN pip install scikit-learn
RUN pip install pandas 
RUN pip install bleurt@https://github.com/google-research/bleurt/archive/b610120347ef22b494b6d69b4316e303f5932516.zip#egg=bleurt

# TODO: We should not use git clone but copy the code from the host
RUN apt update
RUN apt install git -y 
RUN git clone https://github.com/EleutherAI/lm-evaluation-harness
RUN pip install -e lm-evaluation-harness

# Init wandb
#COPY ./wandb /wandb
ENV WANDB_CONFIG_DIR=/wandb

ENV HF_DATASETS_CACHE="/hf_cache/datasets"
ENV HUGGINGFACE_HUB_CACHE="/hf_cache/hub"

# Copy the code
COPY . /code

# Set the working directory
WORKDIR /code
