# NVFuser in PyTorch 2.0

ICYMI:
1. https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41958/
2. https://pytorch.org/blog/Accelerating-Hugging-Face-and-TIMM-models/

`torch.compile()` makes it easy to experiment with different compiler backends to make PyTorch code faster with a single line decorator `torch.compile()`. It works either directly over an nn.Module as a drop-in replacement for `torch.jit.script()` but without requiring you to make any source code changes. We expect this one line code change to provide you with between 30%-2x training time speedups on the vast majority of models that you’re already running.

Results running resnet18 locally. Your results will vary, but hopefully have similar % deltas. AS noted in all the docs, these compiler integrations are dependent on data center GPUs and not gaming GPUs. 


```sh
Using cache found in /workspace/hub/pytorch_vision_v0.10.0
2022-12-05 00:23:21,773 Compiling torch model...
Running Torch Default Model : 100%|█████████████████████████| 1000/1000 [00:08<00:00, 124.02it/s]
Running Torch Compiled Model: 100%|█████████████████████████| 1000/1000 [00:05<00:00, 198.83it/s]
2022-12-05 00:23:48,428 Runtime diff is ~46.355% with a batch size of 32 for 1,000 iterations.
```
