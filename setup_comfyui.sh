#!/bin/bash
# ---------------------------------------------------------
# COMFYUI AUTOMATIC INSTALLER (V100 Optimized)
# ---------------------------------------------------------

WORKSPACE="$HOME/AI"
COMFY_DIR="$WORKSPACE/ComfyUI"
VENV_DIR="$COMFY_DIR/venv"
PIP="$VENV_DIR/bin/pip"
MODELS_DIR="$COMFY_DIR/models/checkpoints"

echo "======================================================="
echo "  🚀 INITIALIZING COMFYUI INSTALLATION "
echo "======================================================="

if [ ! -d "$COMFY_DIR" ]; then
    echo "⚙️ Cloning ComfyUI from GitHub..."
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    git clone https://github.com/comfyanonymous/ComfyUI.git
else
    echo "✅ ComfyUI repository already exists."
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️ Creating isolated venv environment..."
    cd "$COMFY_DIR"
    python3 -m venv "$VENV_DIR"
    $PIP install --upgrade pip
    
    echo "⚙️ Installing PyTorch and dependencies (CUDA 12.4 for V100)..."
    $PIP install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    $PIP install -r requirements.txt
else
    echo "✅ ComfyUI environment already exists!"
fi

echo "======================================================="
echo "  ⚡ DOWNLOADING REALVISXL V4.0 (SDXL BASE MODEL) "
echo "======================================================="
mkdir -p "$MODELS_DIR"
MODEL_PATH="$MODELS_DIR/RealVisXL_V4.0.safetensors"

if [ ! -f "$MODEL_PATH" ]; then
    echo "⏳ Downloading ~7GB model. This will take a minute on Google Cloud..."
    wget -O "$MODEL_PATH" https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors
    echo "✅ Download complete!"
else
    echo "✅ RealVisXL V4.0 already exists!"
fi

echo "======================================================="
echo "  🎉 COMFYUI INSTALLATION COMPLETE! "
echo "======================================================="
