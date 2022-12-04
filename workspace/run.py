import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.io import read_image
from torchvision.models import ResNet18_Weights

from tqdm.auto import tqdm
from time import time
import json
import logging


logging.basicConfig(format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)

torch.set_float32_matmul_precision("high")
torch.manual_seed(1)
batch_size = 32
iterations = 1000
backends = [
    "ansor",
    "aot_cudagraphs",
    "aot_eager",
    "aot_inductor_debug",
    "aot_ts",
    "aot_ts_nvfuser",
    "aot_ts_nvfuser_nodecomps",
    "cudagraphs",
    "cudagraphs_ts",
    "cudagraphs_ts_ofi",
    "dynamo_accuracy_minifier_backend",
    "dynamo_minifier_backend",
    "eager",
    "fx2trt",
    "inductor",
    "ipex",
    "nnc",
    "nnc_ofi",
    "nvprims_aten",
    "nvprims_nvfuser",
    "ofi",
    "onednn",
    "onnx2tensorrt",
    "onnx2tf",
    "onnxrt",
    "onnxrt_cpu",
    "onnxrt_cpu_numpy",
    "onnxrt_cuda",
    "static_runtime",
    "taso",
    "tensorrt",
    "torch2trt",
    "torchxla_trace_once",
    "torchxla_trivial",
    "ts",
    "ts_nvfuser",
    "ts_nvfuser_ofi",
    "tvm",
    "tvm_meta_schedule"]


class Predictor(nn.Module):
    def __init__(self, resnet18):
        super().__init__()
        self.resnet18 = resnet18.eval()
        self.transforms = nn.Sequential(
            T.Resize(
                [
                    256,
                ]
            ),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)


def pct_diff(t1, t2):
    n = abs(t1 - t2)
    d = abs(t1 + t2) / 2
    return round((n / d) * 100, 3)


model = torch.hub.load(
    "pytorch/vision:v0.10.0",
    model="resnet18",
    weights=ResNet18_Weights.DEFAULT,
    progress=False,
)

logger.info("Compiling torch model...")
opt_model = torch.compile(
    model=model,
    backend="ts_nvfuser_ofi",
    # mode="max-autotune",
)


with open("imagenet_class_index.json", "r") as labels_file:
    labels = json.load(labels_file)


default_predictor = Predictor(model).to("cuda")

torch._dynamo.config.suppress_errors = True
opt_predictor = Predictor(opt_model).to("cuda")


size = (800, 600)
dog1 = read_image("dog_1.jpg")
dog1 = T.Resize(size=size)(dog1)
batch = torch.stack([dog1] * batch_size).to("cuda")


default_predictor_result = default_predictor(batch)
for i, pred in enumerate(default_predictor_result):
    logger.debug(f"Prediction for Dog {i + 1}: {labels[str(pred.item())]}")


opt_predictor_result = opt_predictor(batch)
for i, pred in enumerate(opt_predictor_result):
    logger.debug(f"Prediction for Dog {i + 1}: {labels[str(pred.item())]}")

default_start = time()
for i in tqdm(range(iterations), desc="Running Torch Default Model "):
    default_predictor(batch)
default_stop = time()
default_runtime = default_stop - default_start


opt_start = time()
for i in tqdm(range(iterations), desc="Running Torch Compiled Model"):
    opt_predictor(batch)
opt_stop = time()
opt_runtime = opt_stop - opt_start


logger.info(
    "Runtime diff is ~{}% with a batch size of {:,} for {:,} iterations.".format(
        pct_diff(
            default_runtime,
            opt_runtime),
        batch_size,
        iterations))
