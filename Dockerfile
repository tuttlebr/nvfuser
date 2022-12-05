ARG BASE_IMAGE_NAME
FROM $BASE_IMAGE_NAME

RUN pip3 install \
    jupyterlab \
    ipywidgets \
    numpy \
    onnxruntime-gpu \
    onnx \
    tqdm \
    --pre torch[dynamo] \
    torchvision \
    torchaudio \
    --force-reinstall \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu117

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt \
    && cd torch2trt \
    && python setup.py install