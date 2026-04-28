import os
import anthropic
from dotenv import load_dotenv
from pdf_loader import load_pdf_text

# Load .env from the same directory as this file regardless of CWD
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to bike_troubleshooter/.env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client

SYSTEM_PROMPT = """\
You are a bike troubleshooting assistant. Your ONLY knowledge source is the \
owner's manual text provided in this conversation.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the provided manual.
2. If the manual does not contain the answer, respond exactly with:
   "I could not find information about this in the provided manual. \
Please consult a certified technician or your nearest Royal Enfield service centre."
3. Do NOT use any general mechanical knowledge or expertise not present in this document.
4. Reference the page number (e.g. "Page 42") when describing procedures if visible.
5. If an image is provided, describe what you observe and cross-reference with \
the manual — only draw conclusions the manual explicitly supports.
6. Never speculate, infer beyond the text, or fill gaps with outside knowledge.
7. Keep answers concise and practical. Use numbered steps for procedures.\
"""


def ask(
    question: str,
    pdf_path: str,
    api_history: list,
    image_b64: str | None = None,
    image_media_type: str | None = None,
) -> tuple[str, list]:
    """Send a question to Claude and return (answer, updated_history)."""
    pdf_text = load_pdf_text(pdf_path)

    # Build content for this user turn
    user_content: list = []
    if image_b64 and image_media_type:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_b64,
            },
        })
    user_content.append({"type": "text", "text": question})

    if not api_history:
        # First turn: inject the manual text as a cached prefix
        first_content = [
            {
                "type": "text",
                "text": (
                    "OWNER'S MANUAL — use this as your sole knowledge source:\n\n"
                    + pdf_text
                ),
                "cache_control": {"type": "ephemeral"},
            },
            *user_content,
        ]
        messages = [{"role": "user", "content": first_content}]
    else:
        messages = api_history + [{"role": "user", "content": user_content}]

    response = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )

    answer = response.content[0].text
    new_history = messages + [{"role": "assistant", "content": answer}]
    return answer, new_history
