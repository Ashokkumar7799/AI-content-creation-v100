import gradio as gr
import base64
import requests
import json
import time
import io
from PIL import Image, ImageOps

SERVER_URL = "http://localhost:8080/v1/chat/completions"

system_default = """You are Qwen, a world-class AI visual director and prompt engineer.
You craft vivid, hyper-detailed, and photorealistic prompts for image and video generation (Krea-2, SDXL, LTX-Video, Wan 2.2).
You emphasize lighting, camera lenses, material textures, subtle expressions, and natural physical motion."""

def process_and_compress_image(raw_image):
    """
    Accepts any image format (PNG, JPG, JPEG, WEBP, BMP, TIFF, etc.).
    - Fixes EXIF orientation (smartphone cameras)
    - Converts RGBA, CMYK, Palette to clean RGB
    - High-quality Lanczos downsampling if larger than 1536px
    - Compresses in-memory to optimized JPEG (quality=88) to bypass tunnel size limits
    """
    if raw_image is None:
        return None, ""
    try:
        if isinstance(raw_image, str):
            img = Image.open(raw_image)
        else:
            img = raw_image

        # Auto-rotate smartphone photos based on EXIF orientation
        img = ImageOps.exif_transpose(img)

        # Convert to RGB (handles PNG alpha channel, CMYK, Palette mode)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Proportional resize maintaining crisp quality
        max_dim = 1536
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Compress to in-memory JPEG (quality=88 preserves fine skin pores, textures, and details)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=88, optimize=True)
        compressed_bytes = buffer.getvalue()
        b64_str = base64.b64encode(compressed_bytes).decode("utf-8")
        
        info = f"📷 [Attached: {img.size[0]}x{img.size[1]}px, ~{len(compressed_bytes)//1024} KB]\n\n"
        return b64_str, info
    except Exception as e:
        return None, f"⚠️ [Image Processing Error: {str(e)}]\n\n"

def chat_fn(user_text, user_image, history_messages, display_history, system_prompt, temperature, max_tokens, history_mode):
    if not user_text and user_image is None:
        yield display_history, history_messages, "", None, ""
        return

    current_content = []
    user_display = ""

    if user_image is not None:
        b64_img, img_info = process_and_compress_image(user_image)
        if b64_img:
            current_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
            user_display += img_info
        else:
            user_display += img_info

    text_content = user_text.strip() if user_text and user_text.strip() else "Analyze this image in detail and write an optimized visual prompt."
    current_content.append({"type": "text", "text": text_content})
    user_display += text_content

    history_messages.append({"role": "user", "content": current_content})
    display_history.append({"role": "user", "content": user_display})
    display_history.append({"role": "assistant", "content": "⏳ *Starting thinking process...*"})

    dialogue_turns = [m for m in history_messages if m.get("role") != "system"]
    system_msg = {"role": "system", "content": system_prompt}

    if history_mode == "Single Prompt (Fastest)":
        payload_messages = [system_msg, dialogue_turns[-1]]
    elif history_mode == "Last 2 Exchanges":
        payload_messages = [system_msg] + dialogue_turns[-4:]
    else:
        payload_messages = [system_msg] + dialogue_turns

    payload = {
        "messages": payload_messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True
    }

    reasoning_reply = ""
    content_reply = ""
    last_yield_time = 0
    yield_interval = 0.05

    def render_output(r_text, c_text):
        res = ""
        if r_text.strip():
            res += f"💭 **Thinking:**\n```text\n{r_text}\n```\n\n"
        
        if "<think>" in c_text:
            formatted = c_text.replace("<think>", "💭 **Thinking:**\n```text\n")
            if "</think>" in formatted:
                formatted = formatted.replace("</think>", "\n```\n\n🎯 **Final Prompt:**\n")
            res += formatted
        else:
            if r_text.strip() and c_text.strip():
                res += f"🎯 **Final Prompt:**\n{c_text}"
            else:
                res += c_text
        return res

    response = None
    try:
        response = requests.post(SERVER_URL, json=payload, stream=True, timeout=180)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            
                            if 'reasoning_content' in delta and delta['reasoning_content'] is not None:
                                reasoning_reply += str(delta['reasoning_content'])
                            elif 'thought' in delta and delta['thought'] is not None:
                                reasoning_reply += str(delta['thought'])
                            
                            if 'content' in delta and delta['content'] is not None:
                                content_reply += str(delta['content'])
                                
                            current_time = time.time()
                            if current_time - last_yield_time > yield_interval:
                                live_view = render_output(reasoning_reply, content_reply)
                                display_history[-1] = {"role": "assistant", "content": live_view}
                                yield display_history, history_messages, "", None, live_view
                                last_yield_time = current_time
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        error_msg = f"❌ Error communicating with llama-server: {str(e)}\nIs llama-server running on port 8080?"
        display_history[-1] = {"role": "assistant", "content": error_msg}
        yield display_history, history_messages, "", None, error_msg
        return
    finally:
        if response is not None:
            response.close()

    final_view = render_output(reasoning_reply, content_reply)
    display_history[-1] = {"role": "assistant", "content": final_view}
    history_messages.append({"role": "assistant", "content": final_view})
    yield display_history, history_messages, "", None, final_view

def stop_generation():
    """Actively aborts token generation on the GPU in llama-server."""
    try:
        slots_resp = requests.get("http://localhost:8080/slots", timeout=1)
        if slots_resp.status_code == 200:
            slots = slots_resp.json()
            for s in slots:
                slot_id = s.get("id", 0)
                requests.post(f"http://localhost:8080/slots/{slot_id}?action=release", timeout=1)
    except Exception:
        pass

def clear_chat():
    stop_generation()
    return [], [], None, "", ""

with gr.Blocks(title="Qwen3.8-27B Vision Studio") as demo:
    gr.Markdown("# 🎬 Qwen3.8-27B Vision Prompt Studio")
    gr.Markdown("Real-time visual prompt engineering with visible live thinking, 16k context, auto-compression, and GPU Stop.")

    history_messages = gr.State([])

    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="💬 Conversation History", height=520)

            with gr.Row():
                text_input = gr.Textbox(lines=2, placeholder="Describe your scene or follow-up revision...", label="✍️ Message", scale=4)
                with gr.Column(scale=1):
                    send_btn = gr.Button("🚀 Send", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

            with gr.Row():
                clear_btn = gr.Button("🧹 New Chat", variant="secondary")

            latest_output = gr.Textbox(
                label="📋 Latest Output (Select & Copy directly)",
                lines=4,
                interactive=False
            )

        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="📷 Reference Image (PNG, JPG, WEBP, TIFF)")
            
            with gr.Accordion("⚙️ Prompt & History Settings", open=True):
                history_mode = gr.Radio(
                    choices=["Last 2 Exchanges", "Single Prompt (Fastest)", "Full History"],
                    value="Last 2 Exchanges",
                    label="⚡ Memory Mode"
                )
                tokens_slider = gr.Slider(minimum=512, maximum=16384, value=4096, step=512, label="Max Generation Tokens")
                temp_slider = gr.Slider(minimum=0.1, maximum=1.2, value=0.7, step=0.05, label="Temperature")
                system_box = gr.Textbox(lines=3, value=system_default, label="System Persona")

    send_event = send_btn.click(
        fn=chat_fn,
        inputs=[text_input, image_input, history_messages, chatbot, system_box, temp_slider, tokens_slider, history_mode],
        outputs=[chatbot, history_messages, text_input, image_input, latest_output]
    )

    submit_event = text_input.submit(
        fn=chat_fn,
        inputs=[text_input, image_input, history_messages, chatbot, system_box, temp_slider, tokens_slider, history_mode],
        outputs=[chatbot, history_messages, text_input, image_input, latest_output]
    )

    stop_btn.click(
        fn=stop_generation,
        inputs=None,
        outputs=None,
        cancels=[send_event, submit_event]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, history_messages, image_input, text_input, latest_output]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False, max_file_size="50mb")
