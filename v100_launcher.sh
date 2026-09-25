#!/bin/bash

# Define paths
AI_DIR="$HOME/AI"
QWEN_ENV="$AI_DIR/qwen_env"
MODELS_DIR="$AI_DIR/models/llm"

echo "=========================================="
echo " 🚀 V100 AI STUDIO MASTER LAUNCHER"
echo "=========================================="
echo "1. Boot Qwen 3.8 Vision Studio (Chat & Image Mode)"
echo "2. Boot AI Toolkit Web UI (LoRA Training)"
echo "3. Boot Image Generation (ComfyUI - SDXL / Krea 2)"
echo "4. Setup & Download Krea 2 Models & Nodes"
echo "5. Boot Qwen 3.8 Video Director (1 FPS Motion Analyzer -> LTX Prompts)"
echo "9. ❌ Kill all running AI processes and free VRAM"
echo "=========================================="
read -p "Select a launch option (1-9): " option

# Master Kill Switch (Ctrl + C)
trap "echo -e '\n🛑 Stopping all services...'; pkill -f llama-server 2>/dev/null; pkill -f qwen_studio.py 2>/dev/null; pkill -f qwen_video_director.py 2>/dev/null; pkill -f node 2>/dev/null; pkill -f pinggy 2>/dev/null; pkill -f cloudflared 2>/dev/null; pkill -f main.py 2>/dev/null; pkill -P $$ 2>/dev/null; exit 0" SIGINT SIGTERM

# Cleanup old processes
echo "🧹 Cleaning up old connections..."
fuser -k 8080/tcp 2>/dev/null || true
fuser -k 7860/tcp 2>/dev/null || true
fuser -k 7861/tcp 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 8675/tcp 2>/dev/null || true
fuser -k 8188/tcp 2>/dev/null || true
pkill -f llama-server 2>/dev/null || true
pkill -f qwen_studio.py 2>/dev/null || true
pkill -f qwen_video_director.py 2>/dev/null || true
pkill -f node 2>/dev/null || true
pkill -f pinggy 2>/dev/null || true
pkill -f cloudflared 2>/dev/null || true
pkill -f main.py 2>/dev/null || true
sleep 1

# Pinggy auto-reconnect function
start_pinggy() {
    local port=$1
    echo ""
    echo "=========================================="
    echo " 🌐 STARTING PINGGY TUNNEL ON PORT $port"
    echo "=========================================="
    while true; do
        pkill -f pinggy 2>/dev/null || true
        ssh -o StrictHostKeyChecking=no -p 443 -R0:localhost:$port a.pinggy.io > pinggy_url.txt 2>/dev/null &
        sleep 4
        echo -e "\n🟢 YOUR SECURE PUBLIC URL IS:"
        cat pinggy_url.txt
        echo "=========================================="
        sleep 3500
    done
}

if [ "$option" == "1" ]; then
    echo "=========================================="
    echo "🧠 Booting Qwen 3.8 (27B) Backend Server..."
    echo "=========================================="
    
    cd $AI_DIR/llama.cpp/build/bin
    ./llama-server \
      -m $MODELS_DIR/Qwen3.8-27B-Heretic-Q4_K_M.gguf \
      --mmproj $MODELS_DIR/mmproj-Qwen3.8-27B-Q8_0.gguf \
      --host 0.0.0.0 \
      --port 8080 \
      -ngl 50 \
      -t 12 \
      -c 20480 \
      -fa \
      -ctk q8_0 \
      -ctv q8_0 > $AI_DIR/llama_server.log 2>&1 &
      
    echo "⏳ Loading 27B model into memory (waiting for backend)..."
    SERVER_READY=0
    for i in {1..90}; do
        if curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; then
            echo "✅ Qwen 3.8 is 100% loaded and ready in ${i} seconds!"
            SERVER_READY=1
            break
        fi
        sleep 1
    done

    if [ $SERVER_READY -eq 0 ]; then
        echo "❌ Error: llama-server failed to start! Last log lines:"
        tail -n 25 $AI_DIR/llama_server.log
        exit 1
    fi
    
    cd $AI_DIR
    source $QWEN_ENV/bin/activate
    python qwen_studio.py > $AI_DIR/qwen_studio.log 2>&1 &
    
    echo "⏳ Starting UI server on port 7860..."
    for i in {1..30}; do
        if curl -s http://localhost:7860 > /dev/null; then
            echo "✅ Qwen Studio UI is up!"
            break
        fi
        sleep 1
    done
    
    start_pinggy 7860

elif [ "$option" == "2" ]; then
    echo "=========================================="
    echo "⚙️ Booting AI Toolkit Web UI..."
    echo "=========================================="
    
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    cd $AI_DIR/ai-toolkit/ui
    source ../lora_training/bin/activate
    
    echo "⏳ Starting Node.js frontend (port 3000)..."
    npm run dev > $AI_DIR/aitoolkit_ui.log 2>&1 &
    
    for i in {1..60}; do
        if curl -s http://localhost:3000 > /dev/null; then
            echo "✅ UI is up!"
            break
        fi
        sleep 1
    done
    
    start_pinggy 3000

elif [ "$option" == "3" ]; then
    echo "=========================================="
    echo "🎨 Booting ComfyUI (SDXL / Krea 2)..."
    echo "=========================================="
    
    cd $AI_DIR/ComfyUI
    source venv/bin/activate
    
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    echo "⏳ Starting ComfyUI (port 8188 with --lowvram for 16GB VRAM)..."
    python main.py --listen 0.0.0.0 --port 8188 --lowvram --preview-method auto > $AI_DIR/comfyui.log 2>&1 &
    
    echo "⏳ Waiting for ComfyUI to come up on port 8188 (max 60 seconds)..."
    for i in {1..60}; do
        if curl -s http://localhost:8188 > /dev/null; then
            echo "✅ ComfyUI is up!"
            break
        fi
        sleep 1
    done
    
    start_pinggy 8188

elif [ "$option" == "4" ]; then
    echo "=========================================="
    echo "📦 Running Krea 2 Setup & Downloader..."
    echo "=========================================="
    if [ -f "$HOME/setup_krea2.sh" ]; then
        bash "$HOME/setup_krea2.sh"
    elif [ -f "./setup_krea2.sh" ]; then
        bash "./setup_krea2.sh"
    else
        echo "⚠️ setup_krea2.sh not found."
    fi

elif [ "$option" == "5" ]; then
    echo "=========================================="
    echo "🎬 Booting Qwen 3.8 Video Director Studio..."
    echo "=========================================="
    
    # 1. Check if llama-server is already running
    if curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; then
        echo "✅ Qwen 3.8 Backend is already running on port 8080!"
    else
        echo "🧠 Booting Qwen 3.8 Backend Server with Flash Attention & 8-bit KV..."
        fuser -k 8080/tcp 2>/dev/null || true
        pkill -f llama-server 2>/dev/null || true
        sleep 1

        cd $AI_DIR/llama.cpp/build/bin
        ./llama-server \
          -m $MODELS_DIR/Qwen3.8-27B-Heretic-Q4_K_M.gguf \
          --mmproj $MODELS_DIR/mmproj-Qwen3.8-27B-Q8_0.gguf \
          --host 0.0.0.0 \
          --port 8080 \
          -ngl 50 \
          -t 12 \
          -c 20480 \
          -fa \
          -ctk q8_0 \
          -ctv q8_0 > $AI_DIR/llama_server.log 2>&1 &
          
        echo "⏳ Loading 27B model into memory..."
        SERVER_READY=0
        for i in {1..90}; do
            if curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q '"ok"'; then
                echo "✅ Qwen 3.8 is 100% loaded and ready in ${i} seconds!"
                SERVER_READY=1
                break
            fi
            sleep 1
        done

        if [ $SERVER_READY -eq 0 ]; then
            echo "❌ Error: llama-server failed to start! Last log lines:"
            tail -n 25 $AI_DIR/llama_server.log
            exit 1
        fi
    fi
    
    # 2. Boot Video Director UI on port 7861
    fuser -k 7861/tcp 2>/dev/null || true
    pkill -f qwen_video_director.py 2>/dev/null || true
    sleep 1

    cd $AI_DIR
    source $QWEN_ENV/bin/activate
    python -c "import cv2" 2>/dev/null || pip install opencv-python-headless
    python qwen_video_director.py > $AI_DIR/video_director.log 2>&1 &
    
    echo "⏳ Starting Video Director UI on port 7861..."
    for i in {1..30}; do
        if curl -s http://localhost:7861 > /dev/null; then
            echo "✅ Video Director UI is up!"
            break
        fi
        sleep 1
    done
    
    start_pinggy 7861

elif [ "$option" == "9" ]; then
    echo "✅ ALL AI PROCESSES KILLED. RAM IS COMPLETELY FREE."
else
    echo "⚠️ Feature coming soon!"
fi
