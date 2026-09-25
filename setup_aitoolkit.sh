#!/bin/bash
# ---------------------------------------------------------
# AI TOOLKIT (LORA TRAINING) AUTOMATIC INSTALLER (V100 Optimized)
# ---------------------------------------------------------

WORKSPACE="$HOME/AI"
TOOLKIT_DIR="$WORKSPACE/ai-toolkit"
VENV_DIR="$TOOLKIT_DIR/lora_training"
PIP="$VENV_DIR/bin/pip"

echo "======================================================="
echo "  🚀 INITIALIZING AI TOOLKIT (LORA) INSTALLATION "
echo "======================================================="

# 1. Clone the repository
if [ ! -d "$TOOLKIT_DIR" ]; then
    echo "⚙️ Cloning AI Toolkit from GitHub..."
    mkdir -p "$WORKSPACE"
    cd "$WORKSPACE"
    git clone https://github.com/ostris/ai-toolkit.git
else
    echo "✅ AI Toolkit repository already exists."
fi

# 2. Create Isolated Environment
if [ ! -d "$VENV_DIR" ]; then
    echo "⚙️ Creating isolated lora_training environment..."
    cd "$TOOLKIT_DIR"
    python3 -m venv "$VENV_DIR"
    $PIP install --upgrade pip
    
    echo "⚙️ Installing PyTorch and dependencies (CUDA 12.4 for V100)..."
    $PIP install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    $PIP install -r requirements.txt
else
    echo "✅ lora_training environment already exists!"
fi

# 3. Setup Node.js (v22) for the UI
echo "======================================================="
echo "  ⚡ SETTING UP WEB UI (NODE.JS) "
echo "======================================================="
export NVM_DIR="$HOME/.nvm"
if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    echo "⚙️ Installing NVM and Node.js..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
    \. "$NVM_DIR/nvm.sh"
    nvm install 22
else
    \. "$NVM_DIR/nvm.sh"
fi

nvm use 22

echo "⚙️ Installing UI dependencies..."
cd "$TOOLKIT_DIR/ui"
npm install

# 4. Install Cloudflare for tunneling
if ! command -v cloudflared &> /dev/null; then
    echo "⚙️ Installing Cloudflare (cloudflared)..."
    cd /tmp
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
fi

echo "======================================================="
echo "  🎉 AI TOOLKIT INSTALLATION COMPLETE! "
echo "======================================================="
