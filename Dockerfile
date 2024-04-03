# Build stage:
FROM hieupth/mamba:pypy3 AS build

ADD . .
RUN apt-get update && \
    mamba install -c conda-forge conda-pack && \
    mamba env create -f environment.yml && \
    apt-get install ffmpeg libsm6 libxext6  -y

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
FROM python:latest 

# Copy /venv from the previous stage:
COPY --from=build /venv /venv

#
RUN pip install gradio mmcv vietocr pdf2image

WORKDIR /craftdet

COPY ./deploy /craftdet/deploy
COPY ./src /craftdet/src
COPY ./weights /craftdet/weights

CMD ["python", "deploy/deploy_gradio.py"]

