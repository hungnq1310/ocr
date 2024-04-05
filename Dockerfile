# Build stage:
FROM hieupth/mamba:pypy3 AS build

ADD . .
RUN apt-get update && \
    mamba install -c conda-forge conda-pack && \
    mamba env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "craftdet", "/bin/bash", "-c"]

# 
RUN conda-pack -n craftdet -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
# 
RUN /venv/bin/conda-unpack

# Runtime stage:
FROM ubuntu:22.04 AS runtime

# Copy /venv from the previous stage:
COPY --from=build /venv /venv
# Copy
COPY . /ocr
# set workdir
WORKDIR /ocr
# 
RUN apt-get update && apt-get install libgl1-mesa-glx libegl1-mesa libopengl0 -y
# 
SHELL ["/bin/bash", "-c"]
#
RUN source /venv/bin/activate && \
    pip install . && \ 
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
    pip install gradio vietocr pdf2image

ENTRYPOINT source /venv/bin/activate && \
           python deploy/deploy_gradio.py