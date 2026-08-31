import argparse
import os

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="Qwen/Qwen3.5-9B",
    choices=["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"],
)
parser.add_argument(
    "--prompt",
    default="Reply with exactly these three words: GPU test passed",
)
parser.add_argument("--max-new-tokens", type=int, default=64)
args = parser.parse_args()

REVISIONS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
}

revision = REVISIONS[args.model]

print("Model:", args.model)
print("HF_HOME:", os.environ.get("HF_HOME"))
print("Visible GPUs:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available to PyTorch.")

print("Logical GPU 0:", torch.cuda.get_device_name(0))
print("Downloading/loading model...")

processor = AutoProcessor.from_pretrained(
    args.model,
    revision=revision,
    local_files_only=True,
)

model = AutoModelForMultimodalLM.from_pretrained(
    args.model,
    revision=revision,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map=0,
    low_cpu_mem_usage=True,
)
model.eval()

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": args.prompt}],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
response = processor.decode(new_tokens, skip_special_tokens=True)

print("\nResponse:")
print(response)