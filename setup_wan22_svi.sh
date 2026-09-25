#!/bin/bash
# ==============================================================================
# WAN 2.2 14B (SVI 2 PRO + LIGHTX2V 4-STEP GGUF) AUTOMATIC INSTALLER & DOWNLOADER
# Optimized for NVIDIA Tesla V100 (16GB VRAM)
# ==============================================================================

set -e

WORKSPACE="$HOME/AI"
COMFY_DIR="$WORKSPACE/ComfyUI"
VENV_DIR="$COMFY_DIR/venv"
PIP="$VENV_DIR/bin/pip"

DIFFUSION_DIR="$COMFY_DIR/models/diffusion_models"
LORAS_DIR="$COMFY_DIR/models/loras"
VAE_DIR="$COMFY_DIR/models/vae"
TEXT_ENCODER_DIR="$COMFY_DIR/models/text_encoders"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"

echo "=================================================================="
echo "  🚀 WAN 2.2 14B SVI 2 PRO AUTOMATED SETUP & DOWNLOADER"
echo "=================================================================="

# 1. Verify ComfyUI base exists
if [ ! -d "$COMFY_DIR" ]; then
    echo "⚙️ ComfyUI not found. Cloning ComfyUI..."
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd "$COMFY_DIR"
    python3 -m venv "$VENV_DIR"
    $PIP install --upgrade pip
    $PIP install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    $PIP install -r requirements.txt
fi

# 2. Install Required Custom Nodes
echo "=================================================================="
echo "  🔌 Installing / Updating Required Custom Nodes"
echo "=================================================================="
mkdir -p "$CUSTOM_NODES_DIR"
cd "$CUSTOM_NODES_DIR"

install_or_update_node() {
    local repo_url=$1
    local folder_name=$2
    if [ ! -d "$folder_name" ]; then
        echo "⚡ Cloning $folder_name..."
        git clone "$repo_url" "$folder_name"
    else
        echo "✅ $folder_name already installed. Updating..."
        cd "$folder_name" && git pull || true
        cd "$CUSTOM_NODES_DIR"
    fi
    if [ -f "$folder_name/requirements.txt" ]; then
        $PIP install -r "$folder_name/requirements.txt" || true
    fi
}

install_or_update_node "https://github.com/kijai/comfyui-kjnodes.git" "comfyui-kjnodes"
install_or_update_node "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" "ComfyUI-VideoHelperSuite"
install_or_update_node "https://github.com/city96/ComfyUI-GGUF.git" "ComfyUI-GGUF"

# 3. Create Model Directories
mkdir -p "$DIFFUSION_DIR"
mkdir -p "$LORAS_DIR"
mkdir -p "$VAE_DIR"
mkdir -p "$TEXT_ENCODER_DIR"

# 4. Fast Downloader Function (aria2c or wget)
download_file() {
    local url=$1
    local dest_file=$2
    local label=$3

    if [ -f "$dest_file" ]; then
        echo "✅ [Already Exists] $label"
    else
        echo "⏳ Downloading $label..."
        if command -v aria2c &> /dev/null; then
            aria2c -x 16 -s 16 -k 1M -c -o "$(basename "$dest_file")" -d "$(dirname "$dest_file")" "$url"
        else
            wget -c --show-progress -O "$dest_file" "$url"
        fi
        echo "✅ Finished downloading $label"
    fi
}

# Ensure aria2 is installed for maximum download speed
if ! command -v aria2c &> /dev/null; then
    echo "⚡ Installing aria2 for ultra-fast multi-threaded downloading..."
    sudo apt update -y && sudo apt install -y aria2 || true
fi

echo "=================================================================="
echo "  📦 DOWNLOADING WAN 2.2 14B MODELS & LORAS"
echo "=================================================================="

# A. Diffusion Models (GGUF Q4_K_S)
download_file \
    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf" \
    "$DIFFUSION_DIR/wan2.2_i2v_high_noise_14B_Q4_K_S.gguf" \
    "Wan 2.2 I2V 14B High Noise (Q4_K_S GGUF ~7.8GB)"

download_file \
    "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf" \
    "$DIFFUSION_DIR/wan2.2_i2v_low_noise_14B_Q4_K_S.gguf" \
    "Wan 2.2 I2V 14B Low Noise (Q4_K_S GGUF ~7.8GB)"

# B. LoRAs (LightX2V 4-Step Distilled)
download_file \
    "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
    "$LORAS_DIR/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
    "LightX2V High Noise 4-Step LoRA"

download_file \
    "https://huggingface.co/lightx2v/Wan2.2-Distill-Loras/resolve/main/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
    "$LORAS_DIR/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors" \
    "LightX2V Low Noise 4-Step LoRA"

# C. LoRAs (SVI v2 Pro - Stable Video Infinity)
download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors" \
    "$LORAS_DIR/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors" \
    "SVI v2 Pro High Noise Infinite Video LoRA"

download_file \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors" \
    "$LORAS_DIR/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors" \
    "SVI v2 Pro Low Noise Infinite Video LoRA"

# D. Text Encoder (UMT5-XXL FP8 Scaled)
download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$TEXT_ENCODER_DIR/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "UMT5-XXL FP8 Text Encoder (~4.8GB)"

# E. VAE
download_file \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    "$VAE_DIR/wan_2.1_vae.safetensors" \
    "Wan 2.1 / 2.2 VAE"

# 5. Copy Workflow to ComfyUI user workflows
WORKFLOW_SRC="$WORKSPACE/SVI_2_pro_workflow.json"
WORKFLOW_DEST_DIR="$COMFY_DIR/user/default/workflows"
mkdir -p "$WORKFLOW_DEST_DIR"
if [ -f "$WORKFLOW_SRC" ]; then
    cp "$WORKFLOW_SRC" "$WORKFLOW_DEST_DIR/SVI_2_pro_workflow.json"
    echo "✅ Workflow copied to ComfyUI workflows folder: $WORKFLOW_DEST_DIR/SVI_2_pro_workflow.json"
fi

echo "=================================================================="
echo "  🎉 ALL WAN 2.2 14B SVI 2 PRO MODELS DOWNLOADED SUCCESSFULLY!"
echo "=================================================================="
echo "To run, launch ComfyUI (Option 3 in v100_launcher.sh) and load:"
echo "📁 SVI_2_pro_workflow.json"
echo "=================================================================="
