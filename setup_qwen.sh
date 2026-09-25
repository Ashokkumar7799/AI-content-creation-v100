#!/bin/bash
set -e
echo "[1/4] Creating separate Python venv (~/AI/qwen_env)..."
python3 -m venv ~/AI/qwen_env
source ~/AI/qwen_env/bin/activate
echo "[2/4] Installing llama-cpp-python (CUDA 12.4)..."
pip install --upgrade pip
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install gradio pillow huggingface_hub[cli] hf_transfer
echo "[3/4] Setting up Models directory..."
mkdir -p ~/AI/models/llm
export HF_XET_HIGH_PERFORMANCE=1
echo "[4/4] Downloading Vision Projector and Base Model..."
hf download 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF mmproj-Qwen3.8-27B-Q8_0.gguf --local-dir ~/AI/models/llm
hf download 0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF Qwen3.8-27B-Heretic-Q4_K_M.gguf --local-dir ~/AI/models/llm
echo "✅ Setup Complete!"
