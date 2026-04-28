import base64
import io
import os

import anthropic
import gradio as gr
import numpy as np
from PIL import Image

import claude_client
import stt_handler

DEFAULT_PDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "Owner Manual-Himalayan.pdf")
)

# ── Images (base64-embedded so no file-serving needed) ───────────────────────
def _to_data_url(filename: str) -> str:
    from PIL import Image as _Img
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
    fmt = _Img.open(path).format  # detects actual format regardless of extension
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(fmt, "image/png")
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

_BG_DATA_URL   = _to_data_url("bike_bg.png")
_LOGO_DATA_URL = _to_data_url("re_logo.png")

# ── Theme ────────────────────────────────────────────────────────────────────
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="transparent",
    body_text_color="#1A1610",
    block_background_fill="rgba(255,248,230,0.92)",
    block_border_color="rgba(245,184,0,0.40)",
    block_border_width="1px",
    block_radius="12px",
    block_shadow="none",
    block_label_text_size="0.9rem",
    block_label_text_color="#6B5820",
    block_title_text_size="0.9rem",
    block_title_text_color="#6B5820",
    input_background_fill="rgba(255,252,240,0.95)",
    input_border_color="rgba(245,184,0,0.45)",
    input_border_width="1px",
    input_radius="10px",
    input_text_size="1.05rem",
    input_placeholder_color="#B8A060",
    # Primary — golden yellow
    button_primary_background_fill="#F5B800",
    button_primary_background_fill_hover="#D4A000",
    button_primary_text_color="#1A1917",
    button_primary_border_color="transparent",
    button_primary_border_color_hover="transparent",
    # Secondary — dark ghost
    button_secondary_background_fill="rgba(30,25,15,0.70)",
    button_secondary_background_fill_hover="rgba(50,42,20,0.90)",
    button_secondary_border_color="rgba(245,184,0,0.30)",
    button_secondary_border_color_hover="rgba(245,184,0,0.60)",
    button_secondary_text_color="#8A7840",
    button_secondary_text_color_hover="#F5B800",
    button_large_radius="10px",
    button_large_padding="10px 20px",
    chatbot_text_size="1.05rem",
)

CSS = f"""
/* ── Layout ── */
body, html {{ margin: 0; padding: 0; background: transparent !important; font-weight: 600 !important; }}
* {{ font-weight: 600 !important; }}
.gradio-container {{
    max-width: 780px !important;
    margin: 0 auto !important;
    background: transparent !important;
}}
footer {{ display: none !important; }}

/* ── Header card ── */
#app-header {{
    background: rgba(10,8,4,0.80);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 14px;
    border: 1px solid rgba(245,184,0,0.35);
    position: relative;
}}
.re-logo-img {{
    position: absolute;
    top: 14px;
    left: 18px;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    object-fit: cover;
    opacity: 0.92;
}}
.header-text {{
    text-align: center;
    padding: 28px 0 10px;
}}
.re-brand {{
    display: block;
    font-family: "Oswald", sans-serif;
    font-size: 0.6rem;
    letter-spacing: 0.42em;
    color: #F5B800;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.model-name {{
    display: block;
    font-family: "Oswald", sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 10px;
}}
.header-tagline {{
    display: block;
    font-size: 1.05rem;
    font-weight: 700;
    color: #6A5E30;
    letter-spacing: 0.08em;
}}
.mountains {{ display: block; width: 100%; margin-bottom: -3px; }}

/* ── Equal-size mic + image inputs ── */
#mic-row {{ display: flex; align-items: stretch; gap: 8px; }}
#mic-row > * {{ flex: 1 1 0; min-width: 0; }}
#mic-row .block {{ height: 100% !important; }}

/* ── Loading animation ── */
.loading-msg {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0;
}}
.ld-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #F5B800;
    display: inline-block;
    animation: ld-bounce 1.2s infinite ease-in-out;
}}
.ld-dot:nth-child(1) {{ animation-delay: 0s; }}
.ld-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.ld-dot:nth-child(3) {{ animation-delay: 0.4s; }}
@keyframes ld-bounce {{
    0%, 80%, 100% {{ transform: scale(0.7); opacity: 0.4; }}
    40%            {{ transform: scale(1.2); opacity: 1.0; }}
}}
.ld-text {{
    font-size: 0.95rem;
    font-weight: 700;
    color: #6B5820;
    letter-spacing: 0.03em;
    margin-left: 4px;
}}

/* ── Footer ── */
#footer-hint {{
    text-align: center;
    padding: 10px 0 2px;
    font-size: 0.7rem;
    color: #6A5E30;
    letter-spacing: 0.02em;
}}
"""


# ── Logic ─────────────────────────────────────────────────────────────────────
def _image_to_b64(pil_img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


LOADING_HTML = (
    '<div class="loading-msg">'
    '<span class="ld-dot"></span>'
    '<span class="ld-dot"></span>'
    '<span class="ld-dot"></span>'
    '<span class="ld-text">Loading, please wait…</span>'
    '</div>'
)

def submit(text_input, audio_input, image_input, pdf_path, api_history, chat_messages):
    question = (text_input or "").strip()

    if audio_input is not None and not question:
        sample_rate, audio_data = audio_input
        question = stt_handler.transcribe(sample_rate, np.array(audio_data))

    if not question:
        yield api_history, chat_messages, "", None, None
        return

    if not pdf_path or not os.path.exists(pdf_path):
        msgs = chat_messages + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "No valid PDF found — check the path in Manual Settings."},
        ]
        yield api_history, msgs, "", None, None
        return

    image_b64, image_media_type = None, None
    user_display = question
    if image_input is not None:
        image_b64, image_media_type = _image_to_b64(image_input)
        user_display = f"[Image attached]\n{question}"

    # ── Step 1: show user message + loading indicator immediately ──
    loading_msgs = chat_messages + [
        {"role": "user", "content": user_display},
        {"role": "assistant", "content": LOADING_HTML},
    ]
    yield api_history, loading_msgs, "", None, None

    # ── Step 2: call Claude, then replace loading with real answer ──
    try:
        answer, new_history = claude_client.ask(
            question=question,
            pdf_path=pdf_path,
            api_history=api_history,
            image_b64=image_b64,
            image_media_type=image_media_type,
        )
    except anthropic.AuthenticationError:
        answer = "Invalid API key — check your ANTHROPIC_API_KEY in the .env file."
        new_history = api_history
    except anthropic.RateLimitError:
        answer = "Rate limit reached — please wait a moment and try again."
        new_history = api_history
    except anthropic.APIConnectionError:
        answer = "Could not reach the Anthropic API — check your internet connection."
        new_history = api_history
    except Exception as exc:
        answer = f"Something went wrong: {exc}"
        new_history = api_history

    final_msgs = chat_messages + [
        {"role": "user", "content": user_display},
        {"role": "assistant", "content": answer},
    ]
    yield new_history, final_msgs, "", None, None


def clear_chat():
    return [], [], "", None, None


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Himalayan Troubleshooter") as demo:

    gr.HTML('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Oswald:wght@600;700&display=swap">')

    # Fixed full-page background — sits behind everything, not affected by theme CSS
    gr.HTML(f"""
        <div style="
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-image:
                linear-gradient(rgba(255,255,255,0.30), rgba(255,255,255,0.30)),
                url('{_BG_DATA_URL}');
            background-size: cover;
            background-position: center;
            z-index: -1;
            pointer-events: none;
        "></div>
    """)

    gr.HTML(f"""
        <div id="app-header">
            <img src="{_LOGO_DATA_URL}" class="re-logo-img" alt="Royal Enfield"/>
            <div class="header-text">
                <span class="re-brand">Royal &nbsp; Enfield</span>
                <span class="model-name">HIMALAYAN</span>
                <span class="header-tagline">Owner's Manual Assistant</span>
            </div>
            <!-- Layered mountain silhouette -->
            <svg class="mountains" viewBox="0 0 780 110" preserveAspectRatio="none"
                 xmlns="http://www.w3.org/2000/svg">
                <!-- Far range -->
                <path d="M0,110 L0,72 C60,45 100,62 145,28 C175,8 205,18 240,22
                         C275,26 305,48 345,40 C385,32 415,14 455,30
                         C495,46 525,38 560,24 C595,10 625,32 660,38
                         C695,44 730,30 780,42 L780,110 Z"
                      fill="#162A18"/>
                <!-- Mid range — snow peaks -->
                <path d="M0,110 L0,85 C30,78 55,84 80,74 C105,64 122,76 148,65
                         C168,56 185,42 205,58 C222,71 242,62 265,50
                         C285,40 305,52 328,44 C352,35 372,48 400,40
                         C428,32 450,20 475,36 C500,52 525,44 548,34
                         C571,24 598,50 625,56 C652,62 688,46 720,52
                         C745,57 764,52 780,58 L780,110 Z"
                      fill="#1C3420"/>
                <!-- Snow caps — golden to echo the bike's yellow -->
                <path d="M143,29 L152,6 L161,29 Q156,23 152,21 Q148,23 143,29 Z"
                      fill="#F5B800" opacity="0.70"/>
                <path d="M453,31 L462,10 L471,31 Q466,25 462,23 Q458,25 453,31 Z"
                      fill="#F5B800" opacity="0.55"/>
                <path d="M558,25 L566,7 L574,25 Q570,20 566,18 Q562,20 558,25 Z"
                      fill="#F5B800" opacity="0.50"/>
                <!-- Near foothills -->
                <path d="M0,110 L0,96 C25,90 48,94 70,88 C92,82 110,90 135,85
                         C160,80 178,88 205,82 C232,76 255,84 280,78
                         C305,72 328,82 355,76 C382,70 405,80 432,74
                         C460,68 482,78 510,72 C538,66 562,76 590,70
                         C618,64 645,74 672,70 C700,66 735,74 780,72
                         L780,110 Z"
                      fill="#0C1A0E"/>
            </svg>
        </div>
    """)

    chatbot = gr.Chatbot(
        label="",
        show_label=False,
        height=460,
        sanitize_html=False,
        placeholder=(
            "<div style='text-align:center;color:#BCBAB6;"
            "font-size:0.85rem;padding:80px 0'>"
            "Ask a question to get started.</div>"
        ),
    )

    text_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. How do I check the engine oil level?",
        lines=2,
        max_lines=5,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary", scale=3)
        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

    with gr.Row(equal_height=True):
        mic_input = gr.Audio(
            sources=["microphone"],
            type="numpy",
            label="Voice input",
            scale=1,
        )
        image_upload = gr.Image(
            type="pil",
            label="Attach image",
            scale=1,
        )

    gr.HTML(
        "<div id='footer-hint'>"
        "&#9650; Voice transcribed locally &nbsp;&nbsp;"
        "&#9650; Images cross-referenced with manual &nbsp;&nbsp;"
        "&#9650; Powered by Claude"
        "</div>"
    )

    pdf_input = gr.Textbox(value=DEFAULT_PDF, visible=False)

    api_history_state = gr.State([])

    shared_inputs = [text_box, mic_input, image_upload, pdf_input, api_history_state, chatbot]
    shared_outputs = [api_history_state, chatbot, text_box, mic_input, image_upload]

    send_btn.click(fn=submit, inputs=shared_inputs, outputs=shared_outputs)
    text_box.submit(fn=submit, inputs=shared_inputs, outputs=shared_outputs)
    clear_btn.click(fn=clear_chat, outputs=shared_outputs)


if __name__ == "__main__":
    demo.launch(share=True, theme=theme, css=CSS)
