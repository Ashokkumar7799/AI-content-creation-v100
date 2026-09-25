import gradio as gr
import base64
import requests
import json
import time
import io
import cv2
from PIL import Image

SERVER_URL = "http://localhost:8080/v1/chat/completions"

system_default_director = """You are an expert AI Cinematographer and Director specializing in reverse-engineering real-world videos into prompts for LTX-Video 2.3 and Wan 2.2.
You are given a chronological sequence of frames sampled at 1 frame per second (1 fps) from a video.

Analyze the visual sequence across the timeline and output your response in this EXACT structured format:

### 1. 🎬 MOTION & CAMERA CHOREOGRAPHY
- **Subject Action Timeline:** Chronological breakdown of actions from start to finish (body posture, head turns, expressions, gestures).
- **Camera Movement:** Exact trajectory (e.g., slow dolly push-in, subtle handheld breathing, orbital pan, tracking shot) and lens focal length (35mm, 50mm, 85mm).
- **Lighting & Atmosphere:** Lighting direction, shadows, color temperature, and mood.
- **Physics & Micro-Details:** Fabric dynamics, wind fluttering clothing, hair motion, natural blinking and breathing.

### 2. 🎯 MASTER LTX-VIDEO 2.3 PROMPT (Ready to Copy)
Provide a single, continuous, highly descriptive cinematic paragraph specifically tailored for LTX-Video 2.3:
- Begins with shot framing and camera directive ("Cinematic medium shot. The camera slowly pushes in with subtle dolly motion...")
- Describes the subject's chronological actions fluidly
- Details natural physics and fabric movement
- Concludes with cinematic specifications ("photorealistic, 4k cinematic video, smooth 24fps motion, authentic skin texture, shallow depth of field")
- If the subject is your custom character, include 'tarastyles woman'.

### 3. ⚙️ RECOMMENDED COMFYUI SETTINGS
- **Aspect Ratio:** 9:16 vertical (768x1344 or 864x1536)
- **Frame Count:** 65 to 97 frames (~2.5 - 4 seconds)
- **Guidance / CFG:** 3.0 - 3.5
- **I2V Denoise Strength:** 0.85
"""

def extract_keyframes_1fps(video_path, fps_rate=1.0, max_frames=30, resolution_px=512):
    """
    Extracts frames at exactly 1 frame per second (or custom rate).
    Downscales frames to resolution_px (default 512px) for fast token encoding and 7-8 t/s inference.
    """
    if not video_path:
        return [], [], 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / native_fps) if native_fps > 0 else 0

    step = int(native_fps / fps_rate) if native_fps > 0 else 30
    step = max(1, step)

    extracted_images = []
    base64_list = []
    current_frame = 0

    while cap.isOpened() and len(extracted_images) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % step == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)

            # Downscale proportionally to selected resolution (512px by default)
            max_dim = int(resolution_px)
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85, optimize=True)
            b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            extracted_images.append(img)
            base64_list.append(b64_str)

        current_frame += 1

    cap.release()
    return extracted_images, base64_list, duration, native_fps

def analyze_video_fn(video_file, custom_instructions, fps_choice, res_choice, system_prompt, temperature, max_tokens):
    if not video_file:
        yield [], "⚠️ Please upload a video file (.mp4, .mov, .webm) first.", ""
        return

    fps_rate = 1.0 if "1 fps" in fps_choice else 2.0
    res_px = 512 if "512px" in res_choice else 768
    yield [], f"⏳ *Extracting keyframes at {fps_rate} fps ({res_px}px)...*", ""

    try:
        gallery_images, base64_frames, duration, native_fps = extract_keyframes_1fps(video_file, fps_rate=fps_rate, resolution_px=res_px)
    except Exception as e:
        yield [], f"❌ Error reading video: {str(e)}", ""
        return

    if not base64_frames:
        yield [], "❌ Could not extract any frames from the video.", ""
        return

    status_msg = f"✅ Extracted **{len(base64_frames)} frames** (sampled from a {duration:.1f}s video at {fps_rate} fps).\n⏳ *Sending all frames to Qwen 3.8 for motion analysis...*"
    yield gallery_images, status_msg, ""

    # Build the multi-image payload: ALL frames passed at once!
    content_payload = []
    for b64 in base64_frames:
        content_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    user_text = (
        f"This is a chronological sequence of {len(base64_frames)} frames sampled at {fps_rate} fps from a {duration:.1f}-second video.\n"
        f"Additional Creator Instructions: {custom_instructions.strip() if custom_instructions else 'Analyze motion, camera trajectory, and generate exact LTX-Video 2.3 prompt.'}\n\n"
        "Reverse-engineer this entire video sequence into the required structured breakdown and Master LTX-Video 2.3 prompt."
    )
    content_payload.append({"type": "text", "text": user_text})

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_payload}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True
    }

    reasoning_reply = ""
    content_reply = ""
    last_yield_time = 0
    yield_interval = 0.05

    def extract_clean_prompt(c_text):
        if not c_text:
            return ""
        clean = c_text
        if "<think>" in clean and "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        elif "<think>" in clean:
            return ""
        
        if "### 2." in clean:
            parts = clean.split("### 2.")[1]
            if "### 3." in parts:
                prompt_section = parts.split("### 3.")[0]
            else:
                prompt_section = parts
            lines = [l for l in prompt_section.strip().split("\n") if not l.startswith("#") and not l.startswith("🎯")]
            return "\n".join(lines).strip()
        return clean.strip()

    def render_output(r_text, c_text):
        res = ""
        if r_text.strip():
            res += f"💭 **Director Thinking Process:**\n```text\n{r_text}\n```\n\n"
        
        if "<think>" in c_text:
            formatted = c_text.replace("<think>", "💭 **Director Thinking Process:**\n```text\n")
            if "</think>" in formatted:
                formatted = formatted.replace("</think>", "\n```\n\n")
            res += formatted
        else:
            res += c_text
        return res

    response = None
    try:
        response = requests.post(SERVER_URL, json=payload, stream=True, timeout=240)
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
                                live_prompt = extract_clean_prompt(content_reply)
                                yield gallery_images, live_view, live_prompt
                                last_yield_time = current_time
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        err = f"❌ Error communicating with llama-server: {str(e)}\nIs llama-server running on port 8080?"
        yield gallery_images, err, ""
        return
    finally:
        if response is not None:
            response.close()

    final_view = render_output(reasoning_reply, content_reply)
    final_prompt = extract_clean_prompt(content_reply)
    yield gallery_images, final_view, final_prompt

def stop_video_analysis():
    """Immediately stops GPU token generation in llama-server."""
    try:
        slots_resp = requests.get("http://localhost:8080/slots", timeout=1)
        if slots_resp.status_code == 200:
            slots = slots_resp.json()
            for s in slots:
                slot_id = s.get("id", 0)
                requests.post(f"http://localhost:8080/slots/{slot_id}?action=release", timeout=1)
    except Exception:
        pass

def clear_all():
    stop_video_analysis()
    return None, [], "", "", ""

# Standalone UI
with gr.Blocks(title="Qwen 3.8 Video-to-LTX Director Studio") as demo:
    gr.Markdown("# 🎬 Qwen 3.8 Video Director Studio (1 FPS Motion Analyzer)")
    gr.Markdown("Drop any Instagram Reel or video (.mp4). It extracts frames at **1 FPS**, analyzes camera & body motion, and writes the **exact LTX-Video 2.3 prompt** to recreate it!")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="🎥 Upload Reference Video (.mp4, .mov, .webm)")
            
            with gr.Accordion("⚙️ Video Sampling & Director Settings", open=True):
                fps_choice = gr.Radio(
                    choices=["1 fps (Recommended for Reels 5-30s)", "2 fps (Short Clips 2-5s)"],
                    value="1 fps (Recommended for Reels 5-30s)",
                    label="⏱️ Keyframe Sampling Rate"
                )
                res_choice = gr.Radio(
                    choices=["512px (Fast 7-8 t/s - Recommended)", "768px (Ultra Detail ~3 t/s)"],
                    value="512px (Fast 7-8 t/s - Recommended)",
                    label="📐 Keyframe Resolution"
                )
                custom_notes = gr.Textbox(
                    lines=2,
                    placeholder="e.g. Recreate this using tarastyles woman wearing a yellow summer dress in Rome...",
                    label="✍️ Custom Character / Style Overrides (Optional)"
                )
                tokens_slider = gr.Slider(minimum=512, maximum=4096, value=2048, step=256, label="Max Output Tokens")
                temp_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.05, label="Temperature")
                system_box = gr.Textbox(lines=4, value=system_default_director, label="System Director Instructions")

            with gr.Row():
                analyze_btn = gr.Button("🚀 Analyze Video & Generate LTX Prompt", variant="primary", scale=3)
                stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1)

            clear_btn = gr.Button("🧹 Clear Video & Analysis", variant="secondary")

        with gr.Column(scale=1):
            frame_gallery = gr.Gallery(
                label="🎞️ Extracted 1 FPS Keyframe Filmstrip",
                columns=4,
                rows=2,
                height=260,
                object_fit="contain"
            )

            output_markdown = gr.Markdown(
                value="*Upload a video and click 'Analyze Video' to see the frame breakdown and master LTX prompt.*"
            )

            copy_box = gr.Textbox(
                label="📋 Master LTX-Video Prompt (Ready to Copy)",
                lines=5,
                interactive=False
            )

    # Wire actions
    analyze_event = analyze_btn.click(
        fn=analyze_video_fn,
        inputs=[video_input, custom_notes, fps_choice, res_choice, system_box, temp_slider, tokens_slider],
        outputs=[frame_gallery, output_markdown, copy_box]
    )

    stop_btn.click(
        fn=stop_video_analysis,
        inputs=None,
        outputs=None,
        cancels=[analyze_event]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[video_input, frame_gallery, output_markdown, copy_box, custom_notes]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7861, share=False, max_file_size="100mb")
