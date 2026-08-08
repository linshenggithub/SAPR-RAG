#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into the local base model for vLLM evaluation."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    base_model = Path(args.base_model)
    adapter = Path(args.adapter)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base_model={base_model}")
    print(f"[merge] adapter={adapter}")
    print(f"[merge] output_dir={output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload()
    model.save_pretrained(str(output_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(output_dir))
    print("[merge] done")


if __name__ == "__main__":
    main()
