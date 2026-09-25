#!/bin/bash
# ==============================================================================
# WAN 2.2 14B (SVI 2 PRO + LIGHTX2V 4-STEP GGUF) AUTOMATIC INSTALLER & DOWNLOADER
# Optimized for NVIDIA Tesla V100 (16GB VRAM)
# ==============================================================================

set -e

WORKSPACE="$HOME/AI"
COMFY_DIR="$WORKSPACE/ComfyUI"
VENV_DIR="$COMFY_DIR/venv_wan"
PIP="$VENV_DIR/bin/pip"

DIFFUSION_DIR="$COMFY_DIR/models/diffusion_models"
LORAS_DIR="$COMFY_DIR/models/loras"
VAE_DIR="$COMFY_DIR/models/vae"
TEXT_ENCODER_DIR="$COMFY_DIR/models/text_encoders"
CUSTOM_NODES_DIR="$COMFY_DIR/custom_nodes"

echo "=================================================================="
echo "  🚀 WAN 2.2 14B SVI 2 PRO AUTOMATED SETUP & DOWNLOADER"
echo "  🛡️ Isolated Quarantine Environment: venv_wan"
echo "=================================================================="

# 1. Setup Isolated Quarantine Environment (venv_wan)
echo "🐍 Setting up isolated venv_wan to protect Krea 2 / SDXL environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️ Creating fresh virtual environment: $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

if [ ! -f "$VENV_DIR/bin/pip" ]; then
    echo "⚙️ Ensuring pip is installed inside venv_wan..."
    "$VENV_DIR/bin/python" -m ensurepip --upgrade 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python"
fi

echo "⚙️ Ensuring pip, uv, and wheel are up to date..."
$PIP install --upgrade pip uv setuptools wheel

if ! "$VENV_DIR/bin/python" -c "import torch" 2>/dev/null; then
    echo "⚙️ Installing PyTorch (CUDA 12.4 for V100)..."
    $PIP install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

if [ -f "$COMFY_DIR/requirements.txt" ]; then
    echo "⚙️ Installing ComfyUI base requirements..."
    $PIP install -r "$COMFY_DIR/requirements.txt" || true
fi

# Patch comfy_kitchen PEP-585 list[...] schema incompatibility with PyTorch < 2.7
echo "🩹 Applying comfy_kitchen PyTorch schema patch..."
"$VENV_DIR/bin/python" -c "
import glob, os, sys, re

for p in sys.path:
    for f in glob.glob(os.path.join(p, 'comfy_kitchen', '**', '*.py'), recursive=True):
        with open(f, 'r') as fp:
            lines = fp.readlines()
        content = ''.join(lines)
        if 'list[' in content:
            new_lines = [re.sub(r'\blist\[', 'typing.List[', l) for l in lines]
            if not any('import typing' in l for l in lines):
                insert_idx = 0
                for i, l in enumerate(new_lines):
                    if l.strip().startswith('from __future__'):
                        insert_idx = i + 1
                new_lines.insert(insert_idx, 'import typing\n')
            with open(f, 'w') as fp:
                fp.writelines(new_lines)
            print(f'Patched: {f}')
" 2>/dev/null || true



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

# 4. Ultra-Fast Hugging Face CLI Downloader (HF-Transfer Rust Engine)
echo "=================================================================="
echo "  ⚡ CONFIGURING HUGGING FACE TURBO ENGINE (hf-transfer)"
echo "=================================================================="
$PIP install -U "huggingface_hub[hf_transfer]" hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Locate the best available hf CLI binary
if [ -f "$HOME/.local/bin/hf" ]; then
    HF_CMD="$HOME/.local/bin/hf"
elif command -v hf &> /dev/null; then
    HF_CMD="hf"
elif [ -f "$VENV_DIR/bin/hf" ]; then
    HF_CMD="$VENV_DIR/bin/hf"
elif [ -f "$VENV_DIR/bin/huggingface-cli" ]; then
    HF_CMD="$VENV_DIR/bin/huggingface-cli"
elif command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
else
    HF_CMD="huggingface-cli"
fi

download_hf_model() {
    local repo=$1
    local repo_path=$2
    local dest_dir=$3
    local filename=$4

    mkdir -p "$dest_dir"
    local final_path="$dest_dir/$filename"

    if [ ! -f "$final_path" ]; then
        echo "⚡ [HF Turbo] Downloading $filename to $dest_dir/..."
        $HF_CMD download "$repo" "$repo_path" --local-dir "$dest_dir" || {
            echo "⚠️ hf download failed for $filename. Retrying with aria2c/wget..."
            local url_path="${repo_path// /%20}"
            local direct_url="https://huggingface.co/$repo/resolve/main/$url_path"
            if command -v aria2c &> /dev/null; then
                aria2c -x 16 -s 16 -k 1M -c -o "$filename" -d "$dest_dir" "$direct_url"
            else
                wget -c --show-progress -O "$final_path" "$direct_url"
            fi
        }
        
        # Move file if HF nested it into subfolders
        if [ -f "$dest_dir/$repo_path" ] && [ "$dest_dir/$repo_path" != "$final_path" ]; then
            mv "$dest_dir/$repo_path" "$final_path"
        fi
        
        # Cleanup potential empty folder from nested HF path
        local top_folder=$(echo "$repo_path" | cut -d'/' -f1)
        if [ -n "$top_folder" ] && [ "$top_folder" != "$repo_path" ] && [ -d "$dest_dir/$top_folder" ]; then
            rm -rf "$dest_dir/$top_folder"
        fi
        echo "✅ Finished $filename"
    else
        echo "✅ [Already Exists] $filename"
    fi
}

echo "=================================================================="
echo "  📦 DOWNLOADING WAN 2.2 14B MODELS & LORAS (HF TURBO)"
echo "=================================================================="

# A. Diffusion Models (GGUF Q4_K_S)
download_hf_model "QuantStack/Wan2.2-I2V-A14B-GGUF" "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_S.gguf" "$DIFFUSION_DIR" "wan2.2_i2v_high_noise_14B_Q4_K_S.gguf"
download_hf_model "QuantStack/Wan2.2-I2V-A14B-GGUF" "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_S.gguf" "$DIFFUSION_DIR" "wan2.2_i2v_low_noise_14B_Q4_K_S.gguf"

# B. LoRAs (LightX2V 4-Step Distilled)
download_hf_model "lightx2v/Wan2.2-Distill-Loras" "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors" "$LORAS_DIR" "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors"
download_hf_model "lightx2v/Wan2.2-Distill-Loras" "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors" "$LORAS_DIR" "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"

# C. LoRAs (SVI v2 Pro - Stable Video Infinity)
download_hf_model "Kijai/WanVideo_comfy" "LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors" "$LORAS_DIR" "SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors"
download_hf_model "Kijai/WanVideo_comfy" "LoRAs/Stable-Video-Infinity/v2.0/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors" "$LORAS_DIR" "SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors"

# D. Text Encoder (UMT5-XXL FP8 Scaled)
download_hf_model "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$TEXT_ENCODER_DIR" "umt5_xxl_fp8_e4m3fn_scaled.safetensors"

# E. VAE
download_hf_model "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/vae/wan_2.1_vae.safetensors" "$VAE_DIR" "wan_2.1_vae.safetensors"

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
