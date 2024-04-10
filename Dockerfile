# Test with new usage case
FROM python:3.11-slim as compiler
ENV PYTHONUNBUFFERED 1

# Copy 
ADD . /ocr
# # set workdir
WORKDIR /ocr/

RUN python -m venv /venv 

# Enable venv
ENV PATH="/venv/bin:$PATH"

RUN apt-get -y install curl &&\
    pip install . && \ 
    pip install torch torchvision torchaudio && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html && \
    pip install vietocr pdf2image opencv-python

# Runtime stage
FROM python:3.11-slim as runner
# Copy 
COPY ./deploy /ocr
WORKDIR /ocr/

COPY --from=compiler /venv /venv

ENV PATH="/venv/bin:$PATH"

EXPOSE 9000
#
RUN curl -O https://install.tunnelmole.com/xD345/install && sudo bash install
ENTRYPOINT uvicorn gfn.api:app --host 0.0.0.0 --port 9000 & tmole 9000