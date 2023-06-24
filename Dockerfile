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

# Init wandb
COPY ./wandb /wandb
ENV WANDB_CONFIG_DIR=/wandb

# Copy the code
COPY . /code

# Set the working directory
WORKDIR /code

