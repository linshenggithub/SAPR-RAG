#!/usr/bin/env python3
import os
import sys

import torch


def run_matmul(dtype):
    a = torch.randn((16, 4096), device="cuda", dtype=dtype)
    b = torch.randn((4096, 4096), device="cuda", dtype=dtype)
    y = a @ b
    torch.cuda.synchronize()
    print(f"[matmul] {dtype} ok mean={y.float().mean().item():.6f}")


def run_qwen_forward(model_path, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    inputs = tok("hello", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs)
    torch.cuda.synchronize()
    print(f"[qwen_forward] {dtype} ok logits={tuple(out.logits.shape)}")
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    torch.cuda.synchronize()
    print(f"[qwen_generate] {dtype} ok text={tok.decode(gen[0], skip_special_tokens=True)!r}")


def main():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)}")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        run_matmul(dtype)
    if len(sys.argv) > 1:
        run_qwen_forward(sys.argv[1], torch.float16)


if __name__ == "__main__":
    main()
