# Royal Enfield Himalayan — Owner's Manual Assistant

An AI-powered chatbot that answers owner queries strictly from the official Himalayan owner's manual. Supports text, voice, and image input — answers are grounded entirely in the manual, no hallucination.

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.6-orange) ![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

---

## Features

- **Text input** — type any question about the bike
- **Voice input** — speak your question, transcribed locally via Whisper (no external STT API)
- **Image input** — attach a photo of a part/issue, cross-referenced with the manual
- **Strict grounding** — answers only from the manual; declines if information isn't there
- **Prompt caching** — manual is cached on first turn, reducing API cost on follow-up questions
- **Conversation memory** — maintains context across the full session

---

## Architecture

```
bike_troubleshooter/
├── app.py            # Gradio UI — layout, theming, submit/clear logic
├── claude_client.py  # Anthropic SDK — prompt caching, conversation history
├── pdf_loader.py     # PDF text extraction with module-level cache
├── stt_handler.py    # Local Whisper STT — numpy audio input, no ffmpeg required
├── requirements.txt
└── .env              # Not committed — see .env.example
```

**Key design decisions:**
- PDF is extracted as text (not base64) to stay within Claude's token limits
- Whisper accepts numpy arrays directly, bypassing ffmpeg for ARM Mac compatibility
- System prompt + manual text both use `cache_control: ephemeral` to minimise token cost on multi-turn conversations
- Background image and logo are base64-embedded to avoid Gradio 6 file-serving issues

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ojasrao-vibecode/himalayan-troubleshooter.git
cd himalayan-troubleshooter
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

### 4. Run

```bash
python app.py
```

> To share a public link (expires in 72 hours), set `share=True` in the last line of `app.py` before running.

---

## Usage

| Input | How to use |
|-------|-----------|
| Text | Type your question and press Send |
| Voice | Tap mic → record → tap again to stop → press Send |
| Image | Attach a photo of the part or issue alongside your question |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM | Claude Sonnet 4.6 (`anthropic`) |
| UI | Gradio 6 |
| STT | OpenAI Whisper (local, `tiny` model) |
| PDF parsing | pypdf |
| Image handling | Pillow |

---

## Cost Notes

- Claude API is pay-per-use — see [Anthropic pricing](https://www.anthropic.com/pricing)
- Prompt caching reduces cost significantly on multi-turn sessions
- Whisper STT runs fully locally — no additional API cost for voice
