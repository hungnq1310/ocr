# Build stage:
FROM hieupth/mamba:pypy3 AS build

ADD . .
RUN apt-get update && \
    mamba install -c conda-forge conda-pack && \
    mamba env create -f environment.yml 

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "craftdet", "/bin/bash", "-c"]
# 
RUN pip uninstall pillow && \
    pip install vietocr && \
    pip install . 
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
COPY ./deploy /ocr
# set workdir
WORKDIR /ocr
# 
RUN apt-get update && apt-get install libgl1-mesa-glx libegl1-mesa libopengl0 -y
# 
SHELL ["/bin/bash", "-c"]
#
ENTRYPOINT source /venv/bin/activate && \
    python /ocr/deploy/deploy_gradio.py && \
    tail -f /dev/null

# # Test with new usage case
# FROM python:3.11-slim as compiler
# ENV PYTHONUNBUFFERED 1

# # Copy 
# COPY . /ocr
# # # set workdir
# WORKDIR /ocr/

# # RUN python -m venv /venv -> not create venv but instead use the existing one
# COPY --from=build /venv /venv

# # Enable venv
# ENV PATH="/venv/bin:$PATH"

# RUN pip install . && \ 
#     pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
#     pip install gradio vietocr pdf2image

# # Runtime stage
# FROM python:3.11-slim as runner
# # Copy 
# COPY . /ocr
# WORKDIR /ocr/

# COPY --from=compiler /venv /venv

# ENV PATH="/venv/bin:$PATH"

# EXPOSE 7860

# CMD python deploy/deploy_gradio.py