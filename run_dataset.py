import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


# Copy the two real hashes from your existing run_qwen.py.
REVISIONS = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-27B": "fc05daec18b0a78c049392ed2e771dde82bdf654",
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    required=True,
    choices=list(REVISIONS),
)
parser.add_argument(
    "--input",
    default="data/counterfactual_causal_propagation_examples_19.xlsm",
)
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Run only this many unfinished examples.",
)
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

revision = REVISIONS[args.model]
input_path = Path(args.input)

if not input_path.exists():
    raise FileNotFoundError(f"Could not find: {input_path}")

model_short_name = args.model.split("/")[-1].lower()
output_path = Path("results") / f"{model_short_name}-baseline.jsonl"
output_path.parent.mkdir(parents=True, exist_ok=True)

# Read the research data.
data = pd.read_excel(input_path, sheet_name="Final Examples")
required_columns = {
    "Story",
    "Counterfactual Query",
    "Expected Edit",
    "Reasoning",
}

missing = required_columns - set(data.columns)
if missing:
    raise ValueError(f"Missing spreadsheet columns: {sorted(missing)}")

data = data.dropna(subset=["Story", "Counterfactual Query"])


# Discover examples that were already completed.
completed_ids = set()

if output_path.exists():
    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                completed_ids.add(json.loads(line)["example_id"])
            except (json.JSONDecodeError, KeyError):
                pass


examples = []

for row_number, (_, row) in enumerate(data.iterrows(), start=1):
    story = str(row["Story"]).strip()
    query = str(row["Counterfactual Query"]).strip()

    # Stable ID based on the actual example contents.
    digest = hashlib.sha256(
        f"{story}\n{query}".encode("utf-8")
    ).hexdigest()[:12]
    example_id = f"example-{digest}"

    if example_id in completed_ids:
        continue

    examples.append(
        {
            "example_id": example_id,
            "row_number": row_number,
            "story": story,
            "query": query,
            "expectation": str(row["Expected Edit"]).strip(),
            "knowledge_reasoning": str(
                row["Reasoning"]
            ).strip(),
        }
    )

if args.limit is not None:
    examples = examples[: args.limit]

if not examples:
    print("No unfinished examples remain.")
    print("Results:", output_path)
    raise SystemExit(0)


print("Model:", args.model)
print("Revision:", revision)
print("Examples to run:", len(examples))
print("Results:", output_path)
print("Loading model...")

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


for position, example in enumerate(examples, start=1):
    # This is deliberately a minimal baseline prompt.
    # The expectation and reasoning annotations are not included.
    prompt = f"""Story:
{example["story"]}

Instruction:
{example["query"]}

Return only the edited story."""

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
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

    new_tokens = output_ids[
        0, inputs["input_ids"].shape[-1] :
    ]

    response = processor.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    record = {
        **example,
        "model": args.model,
        "revision": revision,
        "condition": "minimal_baseline",
        "prompt": prompt,
        "response": response,
    }

    # Append after every example so progress survives interruptions.
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()

    print(f"\n[{position}/{len(examples)}] Spreadsheet row {example['row_number']}")
    print("MODEL RESPONSE:")
    print(response)
    print("\nEXPECTED CONSEQUENCE:")
    print(example["expectation"])
    print("-" * 70)

print("\nFinished.")
print("Results saved to:", output_path)