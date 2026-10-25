# Quantization tools for OLMoE

Model: `allenai/OLMoE-1B-7B-0924` (64 experts, 8 active per token, about 7B total and 1B active parameters).

Checked against each tool's GitHub repo on 2026-10-25.

## Summary

| Tool | Licence | OLMoE / MoE support | Bit widths | Hardware | Status |
|---|---|---|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | Yes. OLMoE has its own converter (`OlmoeForCausalLM`) and model arch (`olmoe`). | 8, 6, 5, 4, 3, 2 and below (Q8_0, K-quants, IQ-quants) | CPU, Apple Metal, CUDA | Active |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Apache-2.0 | MoE yes (examples for Mixtral, Qwen MoE, DeepSeek, GLM, Granite MoE). No OLMoE example. Output runs in vLLM, which supports OLMoE. | Mainly 8 and 4 (W8A8, W4A16, FP8, FP4); others via custom schemes | NVIDIA GPU | Active |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | Apache-2.0 | Yes. OLMoE is listed in its supported models. | 2, 3, 4, 8 | NVIDIA GPU (some CPU/other back ends) | Active |
| [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) | MIT | Mixtral-style MoE only | 2, 3, 4, 8 | NVIDIA GPU | Archived, replaced by GPTQModel |
| [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) | MIT | Mixtral, Qwen2-MoE, DeepSeek | 4 | NVIDIA GPU | Archived, AWQ now lives in llm-compressor |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) | MIT | Any model loaded by `transformers` (quantizes linear layers on load) | 8, 4 only | NVIDIA GPU (others experimental) | Active |
| [HQQ](https://github.com/dropbox/hqq) | Apache-2.0 | Any model loaded by `transformers` (quantizes linear layers) | 1 to 8 | NVIDIA GPU, CPU | Active |
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | MIT | Yes. Has an `olmoe` model file. | 2, 3, 4, 6, 8 | Apple Silicon only | Active |

## Keeping the router and output layer at high precision

| Tool | Router gates (`mlp.gate`) | Output layer (`lm_head`) |
|---|---|---|
| llama.cpp | Never quantized: `llama-quantize` skips `ffn_gate_inp` tensors | `--output-tensor-type q8_0` or `--leave-output-tensor` |
| llm-compressor | `ignore=["re:.*mlp.gate$"]` | `ignore=["lm_head"]` |
| GPTQModel | Exclude via its dynamic per-module config | Not quantized by default |
| bitsandbytes | `llm_int8_skip_modules=["gate", ...]` | `lm_head` is skipped by default |
| HQQ | Leave it out of the layers you quantize | Same |

## Choice for this team

We use **llama.cpp**:

- It is the only tool here that supports OLMoE directly and covers every bit width the task needs (8 down to 2) in one tool.
- It leaves the router unquantized by default, and one flag keeps the output layer at 8-bit.
- It runs the same on a laptop (Metal), on CPU, and on a Kaggle GPU (CUDA), so the script works from a clean clone anywhere.
- MIT licence, so the quantized files can be shared without restrictions from the tool.

llm-compressor is the better choice if the model will be served with vLLM on NVIDIA GPUs, but it covers fewer bit widths and has no OLMoE example yet.
