# Server Environment

This file records non-sensitive server information for experiment planning and ARIS/Copilot CLI workflows. Do not store passwords, private keys, API tokens, or full credential material here.

## Server B: Current 4 x RTX 3090 Machine

- Role: local execution host for baseline reruns, retrieval preprocessing, badcase classification, ablations, and metric recomputation.
- User: `mayi`
- Project path: `/home/mayi/RAG/agentic-rag-process-optimization`
- External baseline root: `/home/mayi/RAG`
- Notes: use this machine as the default control node for Copilot CLI and ARIS workflows.

## Server A: 3 x RTX 5090 Machine

- SSH alias: `rag-5090`
- Hostname observed by SSH: `expm11`
- IP: `10.249.150.133`
- User: `mayi`
- GPU: `3 x NVIDIA GeForce RTX 5090`
- SSH config: `~/.ssh/config`
- Identity file: `~/.ssh/id_ed25519`
- ReasonRAG path: `/home/mayi/ReasonRAG`
- Role: SAPR-RAG main experiments, candidate trajectory generation, LLM-as-Judge scoring, and reward model training.
- Important: `/home/mayi/ReasonRAG` already contains previous reproduction outputs. Do not overwrite or delete `output/`, `corpus/`, `indexes/`, `dataset/`, or `training_dataset/` unless explicitly requested.

## Quick Checks

```bash
ssh rag-5090 'hostname; whoami; nvidia-smi'
```

Use the SSH alias in ARIS/Copilot CLI experiment requests, for example:

```text
/experiment-bridge "Connect SAPR-RAG experiments to ReasonRAG baseline" — server: rag-5090, gpu: 0, base repo: /home/mayi/ReasonRAG
```
