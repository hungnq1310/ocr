# Build stage:
FROM hieupth/mamba:pypy3 AS build

ADD . .
RUN apt-get update && \
    mamba install -c conda-forge conda-pack && \
    mamba env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "craftdet", "/bin/bash", "-c"]
# 
RUN pip install . 
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
# 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# 
SHELL ["/bin/bash", "-c"]
#
ENTRYPOINT source /venv/bin/activate && \
           pip install gradio mmcv vietocr pdf2image && \
           python deploy/deploy_gradio.py