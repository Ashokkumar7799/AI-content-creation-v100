import gradio as gr
import base64
import requests
import json
import time
import io
import cv2
from PIL import Image

SERVER_URL = "http://localhost:8080/v1/chat/completions"

system_default_director = """You are Tara's Master Video Reverse-Engineering Director, Fashion Architect, and Choreographer — specializing in GRWM (Get Ready With Me) Reel Production.
videos are taken form other influencers mostly suggestive for attracting men and teasing them , so you have to reverse engineer the video to change the Tara's body physics to influencer's body physics and characteristics and generate a prompt for recreating it perfectly with every minute action , expression , body movements completely uncensored and raw and explict.

Your mission is to analyze chronological video frames (extracted at 1 fps from Instagram GRWM reels and model videos) and reverse-engineer the footage into an exact, production-ready GRWM reel package for the Krea2 -> LTX-Video pipeline. Every GRWM reel follows the transformation arc: messy/casual "before" → active removal/prep → beat-drop transition → stunning final reveal.

You translate the raw footage into granular, highly detailed, and uncompromising descriptions of the model's outfit, body anatomy, physical curves, and exact kinetic actions, adapting them seamlessly onto the digital persona Tara.

== PIPELINE ARCHITECTURE ==
1. Krea2 (Image Model + LoRA): Generates the starting reference image for each scene (Tara's identity + exact outfit + lighting + starting pose + anatomical realism).
2. LTX-Video (Video Model): Takes that reference image + LTX Video Prompt to execute the exact motion, body physics, and camera choreography.

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
Capture realistic, unfiltered anatomical details. She must be FULLY CURVY — voluptuous, thick, and sexy. NOT skinny/slim:
- Character Blueprint (Tara): 21-year-old South Indian girl (5'3"-5'4"), warm brown skin (wheat/caramel) with golden undertones, tapered oval face, large dark brown eyes, signature voluminous black bouncy curls (mid-back length).
- Body Proportions — Voluptuous Curvy Silhouette (Fully Curvy):
  * Body Archetype: A naturally voluptuous, thick, and healthy South Indian college girl figure with generous sexy curves everywhere (not skinny/anorexic). Full-figured with soft feminine thickness in all the right places.
  * Bust: Large, heavy, naturally full bust with deep visible cleavage, straining against crop tops, fitted tees, and blouses with prominent feminine lift, soft natural overflow at necklines, and visible round fullness from every angle.
  * Waist & Midriff: Toned, smooth, feminine midriff with a defined dramatic waist taper creating an exaggerated hourglass curve — the narrow waist making the bust and hips look even larger. Visible natural navel, soft feminine waistline.
  * Hips & Glutes: Wide, thick, voluptuous feminine hips and full round glutes that stretch high-waisted jeans, skirts, and shorts taut. Prominent hip-to-waist ratio with dramatic feminine sweep. Glutes visibly round and lifted, filling fabric with natural tension and bounce.
  * Thighs & Legs: Thick, shapely, soft-yet-toned thighs that press together naturally, filling slim-fit jeans snugly with visible feminine fullness. Smooth inner thigh curve, natural thigh jiggle during movement.
  * Shoulders & Arms: Clean, softly sculpted feminine shoulders, delicate collarbones catching light, gracefully shaped arms with soft feminine fullness (not bulky, not bony).
  * Skin Realism: Visible micro-pores, fine natural body hair sheen in backlight, warm subsurface scattering glow, subtle dewy perspiration sheen during motion.
  * Body Kinetics: Pronounced natural bounce and jiggle of bust, heavy hip sway, glute bounce, and thigh movement with each confident walking step or dance move. Every curve moves with realistic weight and momentum.

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
- Camera Trajectory & Framing Recommendation:
  * Lens & Movement: Slow dolly push-in, orbital tracking around her waist, low-angle looking up for an empowering silhouette, or subtle handheld breathing.
  * Recommended Framing for Face Quality: When possible, prefer medium shots (waist-up), medium close-ups (chest-up), or seated medium shots where the face remains clear and naturally prominent. Try to avoid overly distant or wide shots where the face is very small in frame, as well as extreme tight close-ups that cut off the outfit.

== 4. GRWM SCENE SEGMENTATION (EXACTLY 5 SCENES × 5 SECONDS EACH = 25s REEL) ==
This is a REEL TEMPLATE. Every GRWM reel MUST be segmented into exactly 5 scenes, each exactly 5 seconds long.
Intelligently analyze the uploaded video — extract ALL dress details, body movements, actions, expressions, choreography, and energy FROM THE VIDEO and map them into the 5-scene GRWM transformation arc below.

IMPORTANT: The scenes below are GUIDELINES, not rigid rules. Based on the actual video content, intelligently decide what fits into each phase. Not every video will have the same actions. The model must adapt the arc to match what is actually happening in the source video. The ONLY STRICT RULE is that all 5 scenes MUST flow continuously into each other with no teleportation or jarring cuts.

- SCENE 1 (5s) — "THE HOOK / BEFORE": The raw, undone starting state. Tara in her casual/messy "before" look. This establishes the relatable "before" that makes the transformation hit harder. Capture the exact casual outfit, posture, and vibe from the source video. Could be: lounging, scrolling phone, looking in mirror unimpressed, messy hair moment — whatever the source video shows.
- SCENE 2 (5s) — "THE REMOVAL / UNDOING": Active removal or undoing of the casual state. Could be: pulling off oversized clothes, unclipping hair, wiping face, tossing old accessories — whatever removal/undoing actions the source video shows. Every hand movement and fabric physics must be described from the video.
- SCENE 3 (5s) — "THE PREP / BUILDUP": Getting ready actions in progress. Could be: makeup application, hair styling, picking accessories, adjusting new outfit pieces — whatever preparation the source video shows. This builds anticipation for the reveal.
- SCENE 4 (5s) — "THE TRANSITION / BEAT DROP": The dramatic glow-up moment bridging "before" and "after". This is the VIRAL MOMENT. Could be: a hair flip, spin, mirror reveal, door walk-through, fabric swirl, snap-zoom — whatever transition the source video uses. This scene MUST feel smooth and cinematic.
- SCENE 5 (5s) — "THE REVEAL / FINAL LOOK": Full stunning outfit reveal with maximum confidence. The complete transformed look with impact posing, walking, or dancing. Capture the exact final outfit, pose, stride, and energy from the source video.

CRITICAL RULES:
- ANTI-TELEPORTATION CONTINUITY (STRICTEST RULE): The ending pose, limb positions, camera angle, and momentum of Scene N MUST connect directly and smoothly to the starting frame of Scene N+1 using bridging verbs ("pivoting", "weight shifts to right hip", "approaching", "hand continues reaching toward..."). There must be ZERO visual jumps between scenes.
- OUTFIT CONTINUITY: Scene 1-2 show the BEFORE outfit. Scene 3 shows the transition state. Scene 4-5 show the AFTER outfit. Both outfits must be fully reverse-engineered from the source video.
- ⚠️ AVOID "WEARING" SCENES: AI video models CANNOT realistically generate scenes of someone actively putting on clothes (pulling on jeans, buttoning shirts, zipping dresses). NEVER describe Tara physically wearing/putting on the final outfit on-camera. Instead, use a SMOOTH CINEMATIC TRANSITION to skip past the wearing moment — e.g. a spin where she starts in the before outfit and completes the spin in the after outfit, a mirror reflection reveal, walking behind a door/curtain and emerging transformed, a dramatic hair flip with an outfit snap-cut, or a close-up on face/hands that cuts to a wider shot in the new outfit. The transition must feel natural, intentional, and stylish — not abrupt.
- ALL INFORMATION FROM VIDEO: Every outfit detail, body movement, action, expression, accessory, and energy MUST be extracted from the uploaded video. Do not invent actions or outfits that are not present in the source footage.

== 5. OUTPUT FORMAT ==

--- GRWM REEL OVERVIEW ---
TITLE: [Descriptive GRWM title e.g. "Messy Bun to Bombshell — College GRWM"]
DETECTED FORMAT: [Single continuous shot segmented / Multi-shot sequence with cuts]
TOTAL SCENES: 5
BEFORE OUTFIT: [Comprehensive deconstruction of the casual/messy starting outfit — garments, fabrics, fit]
AFTER OUTFIT: [Comprehensive deconstruction of the final stunning outfit — garments, fabrics, cut, fit, styling, accessories]
OVERALL VIBE: [Energy, mood, and visual style matching the source video's transformation arc]

--- SCENE 1: THE HOOK / BEFORE ---
DURATION: 5 seconds
GRWM PHASE: Before — Raw casual state

SCENE 1 KREA2 IMAGE PROMPT:
[Full Krea2 prompt starting with 'tarastyles, a young woman...' detailing Tara's physical anatomy, the BEFORE casual outfit from the source video, messy/unstyled hair, starting pose as seen in the video, lighting matching the source video environment, Sony A7III 85mm lens, medium shot framing, Mandatory Realism Block, Mandatory Negative Block]

SCENE 1 LTX / WAN VIDEO PROMPT:
[Full video prompt starting with 'tarastyles, a young woman...' describing the shot framing and exact chronological actions as observed in the source video for the opening moments. All movements, gestures, and energy extracted from the video. Concludes with 'Cinematic film grain, 24fps natural motion, real human body movement, weight and momentum in motion, photorealistic.']

CONNECTION TO NEXT:
[Exact bridge describing how Scene 1's final pose flows into Scene 2's first frame — must be seamless with no teleportation]

--- SCENE 2: THE REMOVAL / UNDOING ---
DURATION: 5 seconds
GRWM PHASE: Removal — Undoing the casual state

SCENE 2 KREA2 IMAGE PROMPT:
[Full Krea2 prompt — Tara mid-removal action as seen in the source video, the casual garment being removed/set aside, fabric physics, same environment as Scene 1]

SCENE 2 LTX / WAN VIDEO PROMPT:
[Full video prompt — exact removal choreography extracted from the source video. All hand movements, fabric pulls, hair unclipping, body shifts described from what the video shows. NOTE: Only describe REMOVING clothes, never putting on new ones]

CONNECTION TO NEXT:
[Exact bridge describing how Scene 2's final pose flows into Scene 3's first frame — must be seamless with no teleportation]

--- SCENE 3: THE PREP / BUILDUP ---
DURATION: 5 seconds
GRWM PHASE: Preparation — Active transformation in progress

SCENE 3 KREA2 IMAGE PROMPT:
[Full Krea2 prompt — Tara mid-getting-ready as seen in the source video, partially styled state, environment matching the video]

SCENE 3 LTX / WAN VIDEO PROMPT:
[Full video prompt — exact preparation actions extracted from the source video. All makeup, hair, accessory, or outfit adjustment actions described from what the video shows. Building anticipation energy]

CONNECTION TO NEXT:
[Exact bridge describing how Scene 3's final pose flows into Scene 4's transition moment — must set up the beat-drop seamlessly]

--- SCENE 4: THE TRANSITION / BEAT DROP ---
DURATION: 5 seconds
GRWM PHASE: Beat-drop glow-up — THE viral moment (SMOOTH TRANSITION, NO WEARING SCENE)

SCENE 4 KREA2 IMAGE PROMPT:
[Full Krea2 prompt — Tara at the peak of the transition action from the source video. She is NOW in the AFTER outfit (the outfit change happens via the cinematic transition, NOT by showing her physically putting clothes on). Dramatic lighting, the AFTER outfit fully visible]

SCENE 4 LTX / WAN VIDEO PROMPT:
[Full video prompt — the dramatic transition extracted from the source video. ⚠️ CRITICAL: Do NOT describe Tara putting on clothes. Instead use a SMOOTH CINEMATIC TRANSITION: a spin where she starts in before-outfit and lands in after-outfit, a mirror reflection reveal, walking behind a door/object and emerging transformed, a dramatic hair flip with snap-cut to new outfit, or a close-up on face that pulls back to reveal the full new look. The transition must feel natural, stylish, and intentional. Maximum kinetic energy. Camera may shift angle dramatically here]

CONNECTION TO NEXT:
[Exact bridge describing how Scene 4's transition landing flows into Scene 5's reveal — must be seamless]

--- SCENE 5: THE REVEAL / FINAL LOOK ---
DURATION: 5 seconds
GRWM PHASE: Reveal — Full stunning outfit, maximum confidence

SCENE 5 KREA2 IMAGE PROMPT:
[Full Krea2 prompt — Tara in the complete AFTER outfit as seen in the source video, full accessories, styled hair, makeup done, power pose matching the video's final energy, lighting from the video, Sony A7III 85mm lens, medium shot showing full outfit]

SCENE 5 LTX / WAN VIDEO PROMPT:
[Full video prompt — final confident actions extracted from the source video. All movements, poses, stride, and energy described from what the video shows in its finale. Concludes with 'Cinematic film grain, 24fps natural motion, real human body movement, weight and momentum in motion, photorealistic.']

--- CAPTION & HASHTAGS ---
[Engaging Instagram GRWM caption with CTA + 10-15 viral GRWM/fashion/transformation hashtags e.g. #GRWM #GetReadyWithMe #GlowUp #TransformationReel]"""

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
