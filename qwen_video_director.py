import gradio as gr
import base64
import requests
import json
import time
import io
import cv2
from PIL import Image

SERVER_URL = "http://localhost:8080/v1/chat/completions"

system_default_director = """You are Tara's Master Video Reverse-Engineering Director, Fashion Architect, and Choreographer.
Your mission is to analyze chronological video frames (extracted at 1 fps from Instagram reels and model videos) and reverse-engineer the footage into an exact, production-ready AI video generation package for the Krea2 -> LTX-Video pipeline.

You translate the raw footage into granular, highly detailed, and uncompromising descriptions of the model's outfit, body anatomy, physical curves, and exact kinetic actions, adapting them seamlessly onto the digital persona Tara.

== PIPELINE ARCHITECTURE ==
1. Krea2 (Image Model + LoRA): Generates the starting reference image for each scene (Tara's identity + exact outfit + lighting + starting pose + anatomical realism).
2. LTX-Video (Video Model): Takes that reference image + LTX Video Prompt to execute the exact motion, body physics, and camera choreography.
3. Audio: Telugu voiceover (casual, flirty, street-smart college girl tone) + SFX + background music.

== 1. DEEP OUTFIT & FASHION REVERSE-ENGINEERING ==
Do not give generic descriptions. Break down the outfit with designer-level precision:
- Exact Garment Anatomy:
  * Tops/Dresses: Specific cut (corset, plunge neckline, halter, sweetheart, square neck, backless, cowl neck, tube top, wrap crop top), strap width, cup contouring, sheer panels, underwire stitching, low back or cross-tie details.
  * Bottoms/Skirts: High-rise vs low-waist, fit (bodycon, flared, wrap, split-thigh up to hip), skirt hemline length (micro, mini, midi), side ruching, drawstring gathers, denim distressing, pockets.
  * Ethnic / Saree / Fusion: Saree drape style (pleats over waist, sheer georgette or organza showing midriff, satin petticoat silhouette), sleeveless blouse cut, deep back neck with dori ties.
- Fabric Physics & Body Cling:
  * Material Texture: Wet-look satin sheen, ribbed knit ridges, stretchy spandex, semi-sheer chiffon, distressed denim, buttery leather.
  * Tension & Draping: How fabric pulls taut across the bust and fuller hips, creases at the waistline, gathers at the hip curve, or flutters and swirls around the legs during movement.

== 2. MINUTE BODY ANATOMY & CURVE PHYSICS (MAPPED ONTO TARA) ==
Capture realistic, unfiltered anatomical details. Never sanitize or turn her into an artificial mannequin:
- Character Blueprint (Tara): 21-year-old South Indian girl (5'3"-5'4"), warm brown skin (wheat/caramel) with golden undertones, tapered oval face, large dark brown eyes, signature voluminous black bouncy curls (mid-back length).
- Body Proportions & Soft Curves:
  * Fuller, natural Indian college girl build: Soft fleshy upper arms, wider hips with natural pear-shaped contour, full thighs that touch slightly when standing, soft natural waistline (NOT an unnatural flat or bony stomach).
  * Torso & Midriff Dynamics: Natural soft tissue folds visible when twisting or bending, visible navel, subtle breathing rise-and-fall of chest.
  * Extremities & Joints: Collarbones catching light, soft curve of the shoulders, realistic slight hyperpigmentation (darker tone at elbows, knees, and knuckles).
  * Skin Realism: Visible micro-pores, fine natural body hair sheen in backlight, warm subsurface scattering glow, subtle dewy perspiration sheen during motion.
  * Body Kinetics: Natural soft bounce and sway of hips, thighs, and body curves with each step or dance move.

== 3. MODEL ACTIONS, POSING & CHOREOGRAPHY FIDELITY ==
Analyze every second of the model's performance with exact mechanical precision:
- Micro-Posing & Seductive Dynamics:
  * Gaze & Head: Locking direct eye contact into the camera lens with a playful smirk, lifting chin, tilting head to let curls fall over one shoulder, biting lower lip, slow over-the-shoulder glance.
  * Hand Interactions: Sliding fingertips along collarbones or neckline, resting hands on fuller hips, running fingers through curly hair, adjusting an outfit strap or waistline, swinging arms casually.
- Kinetics, Dance & Footwork:
  * Lower Body & Stride: Exaggerated catwalk hip sway, crossing steps, heel taps, pivots, shifting weight from one leg to the other, popping one knee forward while resting weight on opposite hip.
  * Torso & Upper Body: Rhythmic chest pops, pelvic rolls, torso twisting, fluid shoulder dips matching the beat.
- Secondary Momentum:
  * Hair Physics: Heavy curly hair whipping around shoulders, bouncy inertia following head turns.
  * Clothing Dynamics: Skirt hem swirling outward with pivots, straps shifting over shoulders, fabric stretching and relaxing with movement.
- Camera Trajectory:
  * Lens & Movement: Slow dolly push-in, orbital tracking around her waist, low-angle looking up for an empowering silhouette, or subtle handheld breathing.

== 4. INTELLIGENT SCENE SEGMENTATION (MAX 5 SCENES) ==
- If the video contains hard camera cuts: Split into distinct scenes at each cut point.
- If the video is one continuous take (e.g. 15s dance/strut reel): Intelligently segment it into 3 to 5 second choreographic phases (maximum 5 scenes total).
- ANTI-TELEPORTATION CONTINUITY: The ending pose, limb positions, and momentum of Scene N MUST connect directly to the starting frame of Scene N+1 using bridging verbs ("pivoting", "weight shifts to right hip", "approaching").

== 5. OUTPUT FORMAT ==

--- REEL OVERVIEW ---
TITLE: [Descriptive title of the reel]
DETECTED FORMAT: [Single continuous shot segmented / Multi-shot sequence with cuts]
TOTAL SCENES: [1 to 5]
OUTFIT BLUEPRINT: [Comprehensive deconstruction of garments, fabrics, cut, fit, and styling]
OVERALL VIBE & MUSIC: [BPM, genre, rhythm matching the movement]

--- SCENE 1 ---
DURATION: [X seconds / Frame range]
TYPE: [ACTION / DANCE / FASHION STRUT / TALKING]

SCENE 1 KREA2 IMAGE PROMPT:
[Full Krea2 prompt starting with 'tarastyles, a young woman...' detailing Tara's physical anatomy, exact outfit cuts, fabric texture, starting pose, lighting, Sony A7III 85mm lens, Mandatory Realism Block, Mandatory Negative Block]

SCENE 1 LTX VIDEO PROMPT:
[Full LTX prompt starting with 'tarastyles, a young woman...' describing the exact chronological actions, footwork, hip sway, body curve kinetics, fabric/hair momentum, and camera movement. Concludes with 'Cinematic film grain, 24fps natural motion, real human body movement, weight and momentum in motion, photorealistic.']

CONNECTION TO NEXT:
[How the final pose and body momentum of Scene 1 directly carry into the first frame of Scene 2]

SCENE 1 AUDIO:
Telugu: "[Telugu script]"
Romanized: "[Romanized Telugu]"
English: "[English translation]"
SFX & Music Cue: [Timing of beat drops, heels clicking, fabric swish]

[Repeat for SCENE 2, SCENE 3... up to MAX 5 SCENES]

--- CAPTION & HASHTAGS ---
[Engaging Instagram caption with CTA, playful Telugu/English slang + 10-15 viral fashion/reels hashtags]"""

def extract_keyframes_1fps(video_path, fps_rate=1.0, max_frames=30, resolution_px=512):
    """
    Extracts frames at exactly 1 frame per second (or custom rate).
    Downscales frames to resolution_px (default 512px) for fast token encoding and high tokens/sec.
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

def extract_clean_prompt(c_text):
    if not c_text:
        return ""
    clean = c_text
    if "<think>" in clean and "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    elif "<think>" in clean:
        return ""
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

def director_fn(user_text, video_file, history_messages, display_history, video_cache, fps_choice, res_choice, history_mode, system_prompt, temperature, max_tokens):
    if not user_text and video_file is None and not history_messages:
        yield display_history, history_messages, "", [], "", video_cache
        return

    fps_rate = 1.0 if "1 fps" in fps_choice else 2.0
    res_px = 512 if "512px" in res_choice else 768
    
    current_content = []
    user_display = ""
    gallery_images = video_cache.get("gallery", []) if video_cache else []

    is_new_video = False
    if video_file is not None:
        cached_path = video_cache.get("path") if video_cache else None
        if cached_path != video_file:
            is_new_video = True

    if is_new_video:
        display_history.append({"role": "user", "content": f"🎥 *Extracting keyframes at {fps_rate} fps ({res_px}px)...*"})
        display_history.append({"role": "assistant", "content": "⏳ *Processing video frames...*"})
        yield display_history, history_messages, "", gallery_images, "", video_cache
        display_history.pop()
        display_history.pop()

        try:
            gallery_images, base64_frames, duration, native_fps = extract_keyframes_1fps(video_file, fps_rate=fps_rate, resolution_px=res_px)
            video_cache = {"path": video_file, "frames": base64_frames, "gallery": gallery_images}
        except Exception as e:
            err_msg = f"❌ Error extracting video frames: {str(e)}"
            display_history.append({"role": "user", "content": "🎥 Video upload"})
            display_history.append({"role": "assistant", "content": err_msg})
            yield display_history, history_messages, "", [], "", video_cache
            return

        for b64 in base64_frames:
            current_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

        user_prompt_text = (
            f"This is a chronological sequence of {len(base64_frames)} frames sampled at {fps_rate} fps from a {duration:.1f}-second video.\n"
            f"Additional Creator Instructions: {user_text.strip() if user_text and user_text.strip() else 'Reverse-engineer the exact motion, choreography, footwork, physics, and camera path.'}\n\n"
            "Reverse-engineer this entire video sequence into the required production package (up to 5 scenes max) with exact Krea2 reference image prompts and micro-choreographed LTX-Video prompts featuring Tara."
        )
        current_content.append({"type": "text", "text": user_prompt_text})
        user_display = f"🎥 **Analyzed Video:** {len(base64_frames)} frames ({duration:.1f}s at {fps_rate} fps)\n\n"
        if user_text and user_text.strip():
            user_display += f"✍️ **Instructions:** {user_text.strip()}"
    else:
        text_content = user_text.strip() if user_text and user_text.strip() else "Refine the master LTX-Video prompt based on previous analysis."
        current_content.append({"type": "text", "text": text_content})
        user_display = text_content

    history_messages.append({"role": "user", "content": current_content})
    display_history.append({"role": "user", "content": user_display})
    display_history.append({"role": "assistant", "content": "⏳ *Starting director thinking process...*"})
    yield display_history, history_messages, "", gallery_images, "", video_cache

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

    response = None
    try:
        response = requests.post(SERVER_URL, json=payload, stream=True, timeout=300)
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
                                display_history[-1] = {"role": "assistant", "content": live_view}
                                yield display_history, history_messages, "", gallery_images, live_prompt, video_cache
                                last_yield_time = current_time
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        err_msg = f"❌ Error communicating with llama-server: {str(e)}\nIs llama-server running on port 8080?"
        display_history[-1] = {"role": "assistant", "content": err_msg}
        yield display_history, history_messages, "", gallery_images, "", video_cache
        return
    finally:
        if response is not None:
            response.close()

    final_view = render_output(reasoning_reply, content_reply)
    final_prompt = extract_clean_prompt(content_reply)
    display_history[-1] = {"role": "assistant", "content": final_view}
    history_messages.append({"role": "assistant", "content": final_view})
    yield display_history, history_messages, "", gallery_images, final_prompt, video_cache

def stop_video_analysis():
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
    return [], [], "", None, [], "", {"path": None, "frames": [], "gallery": []}

with gr.Blocks(title="Qwen 3.8 Video Director Studio") as demo:
    gr.Markdown("# 🎬 Qwen 3.8 Video Director Studio (1 FPS Motion Analyzer -> LTX Prompts)")
    gr.Markdown("Drop any Instagram Reel or video (.mp4). Analyzes camera motion, actor actions, and outputs ready-to-copy LTX-Video 2.3 prompts.")

    history_messages = gr.State([])
    video_cache = gr.State({"path": None, "frames": [], "gallery": []})

    with gr.Row():
        # LEFT COLUMN: Generation, Interactive Revisions, and Master Copy Box (scale=5)
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="🎬 Director Analysis & Revisions", height=520)

            with gr.Row():
                text_input = gr.Textbox(
                    lines=2,
                    placeholder="Enter custom character notes (e.g. tarastyles in yellow dress) or follow-up revisions...",
                    label="✍️ Instructions / Follow-up Revisions",
                    scale=4
                )
                with gr.Column(scale=1):
                    analyze_btn = gr.Button("🚀 Analyze / Send", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

            with gr.Row():
                clear_btn = gr.Button("🧹 New Analysis / Clear", variant="secondary")

            latest_output = gr.Textbox(
                label="📋 Complete Production Package & LTX Prompts (Select & Copy)",
                lines=6,
                interactive=False
            )

        # RIGHT COLUMN: Video Upload, Filmstrip Gallery, and Settings (scale=2)
        with gr.Column(scale=2):
            video_input = gr.Video(label="🎥 Reference Video (.mp4, .mov, .webm)")
            
            frame_gallery = gr.Gallery(
                label="🎞️ Extracted 1 FPS Keyframe Filmstrip",
                columns=3,
                rows=2,
                height=220,
                object_fit="contain"
            )

            with gr.Accordion("⚙️ Video Sampling & Director Settings", open=True):
                history_mode = gr.Radio(
                    choices=["Last 2 Exchanges", "Single Prompt (Fastest)", "Full History"],
                    value="Last 2 Exchanges",
                    label="⚡ Memory Mode"
                )
                fps_choice = gr.Radio(
                    choices=["1 fps (Recommended for Reels 5-30s)", "2 fps (Short Clips 2-5s)"],
                    value="1 fps (Recommended for Reels 5-30s)",
                    label="⏱️ Keyframe Sampling Rate"
                )
                res_choice = gr.Radio(
                    choices=["512px (Fast 7-8 t/s - Recommended)", "768px (Ultra Detail ~4 t/s)"],
                    value="512px (Fast 7-8 t/s - Recommended)",
                    label="📐 Keyframe Resolution"
                )
                tokens_slider = gr.Slider(
                    minimum=512,
                    maximum=16384,
                    value=16000,
                    step=512,
                    label="Max Generation Tokens"
                )
                temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.2,
                    value=0.6,
                    step=0.05,
                    label="Temperature"
                )
                system_box = gr.Textbox(
                    lines=4,
                    value=system_default_director,
                    label="System Director Instructions"
                )

    analyze_event = analyze_btn.click(
        fn=director_fn,
        inputs=[text_input, video_input, history_messages, chatbot, video_cache, fps_choice, res_choice, history_mode, system_box, temp_slider, tokens_slider],
        outputs=[chatbot, history_messages, text_input, frame_gallery, latest_output, video_cache]
    )

    submit_event = text_input.submit(
        fn=director_fn,
        inputs=[text_input, video_input, history_messages, chatbot, video_cache, fps_choice, res_choice, history_mode, system_box, temp_slider, tokens_slider],
        outputs=[chatbot, history_messages, text_input, frame_gallery, latest_output, video_cache]
    )

    stop_btn.click(
        fn=stop_video_analysis,
        inputs=None,
        outputs=None,
        cancels=[analyze_event, submit_event]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[chatbot, history_messages, text_input, video_input, frame_gallery, latest_output, video_cache]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7861, share=False, max_file_size="100mb")
