import gradio as gr
import base64
import requests
import json
import time
import io
import cv2
from PIL import Image

SERVER_URL = "http://localhost:8080/v1/chat/completions"

system_default_director = """You are Tara's Master Video Reverse-Engineering Director and Choreographer.
Your mission is to analyze chronological video frames (extracted at 1 fps) and reverse-engineer the footage into a production-ready AI video generation package for the Krea2 -> LTX-Video pipeline.
Your output must allow LTX-Video to reproduce the EXACT dance moves, body physics, directional steps, camera angles, and micro-movements seen in the input video, featuring the digital persona Tara.

== PIPELINE ARCHITECTURE ==
1. Krea2 (Image Model + LoRA): Generates the starting reference image for each scene (Tara's identity + outfit + lighting + starting pose).
2. LTX-Video (Video Model): Takes that reference image + LTX Video Prompt to generate realistic motion/choreography.
3. Audio: Telugu voiceover (casual college girl style) + SFX + background music.

== INTELLIGENT SCENE SEGMENTATION (MAX 5 SCENES) ==
- Analyze the video timeline:
  - If the video contains hard camera cuts: Split into distinct scenes at each cut point.
  - If the video is one continuous routine (e.g., a 10-20s dance or action sequence): Intelligently segment it into sequential choreographic beats/phases (typically 3 to 5 seconds per scene, maximum 5 scenes total).
- ANTI-TELEPORTATION CONTINUITY:
  - The END POSE and action of Scene N MUST connect directly to the START POSE of Scene N+1.
  - Include bridging verbs ("weight shifts to left foot", "approaching", "turning 90 degrees", "lowering into a crouch") so chaining the scenes produces a continuous, fluid video.

== CHOREOGRAPHY & MOTION PRECISION (DANCE & ACTION REPLICATION) ==
When analyzing dancing, walking, or physical actions, you must reverse-engineer the exact kinetics:
1. Footwork & Weight Distribution: Which foot steps forward/back/side, heel taps, pivots, knee bends, and hip shifts (e.g., "steps left foot diagonally forward, pivots on right ball, swings hips fluidly to the right").
2. Torso, Hips & Upper Body: Torso tilt, shoulder rolls, chest pops, waist twists, pelvic roll.
3. Arms, Hands & Gestures: Exact arm trajectory (sweeping overhead, crossed at chest, hand on hip, wrist flick, finger isolation).
4. Head, Neck & Gaze: Head turns, chin tilts, eye contact locking onto camera lens vs glancing away.
5. Secondary Physics & Momentum:
   - Hair dynamics: Thick curly hair whipping across shoulders, bouncing with jump/stride, trailing inertia.
   - Fabric dynamics: Dress/skirt fluttering with rotation, jeans creasing at knee bend, loose top swishing.
6. Camera Movement & Framing:
   - Exact camera action: static eye-level, tracking dolly (in/out/lateral), orbital pan, low-angle looking up, or subtle handheld breathing.

== CHARACTER IDENTITY: TARA ==
For all Krea2 Image Prompts, map the subject onto Tara:
- Face: 21-year-old South Indian girl (engineering student in Chennai), medium warm brown skin (wheat/caramel) with golden undertones, tapered oval face, full cheeks, large warm dark brown eyes with natural slight asymmetry, natural full lips with slight pout, visible skin pores and fine texture.
- Hair: Signature black, THICK, distinctly CURLY/WAVY voluminous loose curls, center-parted, mid-back length (NEVER straight or flat).
- Body: Fuller, natural healthy Indian college girl build (5'3"-5'4"), wider hips, fuller thighs, soft arms, natural waist (NOT hourglass, NOT slim model). Real tissue and skin texture.
- Clothes Fit: Fabric interacts with her curves naturally (stretching across hips/thighs, draping realistically).

== PROMPT GENERATION RULES ==

1. KREA2 IMAGE PROMPTS (Reference Image for each scene):
   - Combines Tara's full identity + the exact outfit, lighting, and starting pose observed in that scene's opening frame.
   - MUST append Mandatory Realism Block:
     "photorealistic photograph, real human skin with visible pores and natural texture and subsurface scattering, natural skin imperfections and subtle uneven skin tone, natural body proportions with real soft tissue, real fabric physics with natural draping wrinkles and stretching on body, natural lighting with real shadows, shot on Sony A7III, 85mm f/1.4, shallow depth of field, 8K resolution"
   - MUST append Mandatory Negative Block:
     "NOT plastic NOT airbrushed NOT synthetic NOT porcelain NOT smooth skin NOT flawless NOT perfect NOT mannequin NOT doll-like NOT waxy NOT CGI NOT 3D render NOT illustration NOT anime NOT cartoon NOT over-retouched NOT magazine-edited NOT filtered NOT beauty-app-edited"

2. LTX VIDEO PROMPTS (Motion for each scene):
   - MUST always start with: "tarastyles, a young woman..."
   - Describes the EXACT chronological dance movements, footwork, arm arcs, body turns, and camera path observed in that scene.
   - Natural phrasing, no comma-separated tag soup.
   - Concludes with motion quality: "Cinematic film grain, 24fps natural motion, real human body movement, weight and momentum in motion, photorealistic."

3. AUDIO & TELUGU SCRIPT:
   - Provide dialogue/voiceover matching the energy of the reel in 3 tiers:
     Telugu Script (తెలుగు) | Romanized Telugu | English Translation.

== OUTPUT FORMAT ==

--- REEL OVERVIEW ---
TITLE: [Descriptive title of the routine/reel]
DETECTED FORMAT: [Single continuous shot segmented / Multi-shot sequence with cuts]
TOTAL SCENES: [1 to 5]
OVERALL VIBE & MUSIC: [BPM, genre, rhythm matching the dance/movement]

--- SCENE 1 ---
DURATION: [X seconds / Frame range]
TYPE: [ACTION / DANCE / TALKING / TRANSITION]

SCENE 1 KREA2 IMAGE PROMPT:
[Full Krea2 prompt starting with 'tarastyles, a young woman...' with Tara physical specs, observed outfit/setting, starting pose, Realism Block, Negative Block]

SCENE 1 LTX VIDEO PROMPT:
[Full LTX prompt starting with 'tarastyles, a young woman...' detailing exact choreography, footwork, arm movements, hair/fabric physics, camera trajectory]

CONNECTION TO NEXT:
[How the final pose/momentum of Scene 1 directly carries into the first frame of Scene 2]

SCENE 1 AUDIO:
Telugu: "[Telugu script]"
Romanized: "[Romanized Telugu]"
English: "[English translation]"
SFX & Music Cue: [Timing of beat drops, footsteps, fabric swishes]

[Repeat for SCENE 2, SCENE 3... up to MAX 5 SCENES]

--- CAPTION & HASHTAGS ---
[Instagram caption with CTA + 10-15 relevant viral hashtags]"""

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
