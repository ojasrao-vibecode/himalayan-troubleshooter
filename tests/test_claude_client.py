from unittest.mock import MagicMock, patch
import claude_client


def _mock_response(text: str):
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


def test_first_turn_injects_pdf_text():
    with patch("claude_client._get_client") as mock_get_client, \
         patch("claude_client.load_pdf_text", return_value="MANUAL CONTENT"):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response("Answer")
        mock_get_client.return_value = mock_client

        claude_client.ask("What is oil capacity?", "fake.pdf", api_history=[])

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        first_content = messages[0]["content"]
        combined = " ".join(b["text"] for b in first_content if b.get("type") == "text")
        assert "MANUAL CONTENT" in combined


def test_subsequent_turns_append_to_history():
    existing_history = [
        {"role": "user", "content": [{"type": "text", "text": "First question"}]},
        {"role": "assistant", "content": "First answer"},
    ]
    with patch("claude_client._get_client") as mock_get_client, \
         patch("claude_client.load_pdf_text", return_value="MANUAL"):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response("Second answer")
        mock_get_client.return_value = mock_client

        claude_client.ask("Second question", "fake.pdf", api_history=existing_history)

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0] == existing_history[0]
        assert messages[1] == existing_history[1]
        assert messages[2]["role"] == "user"


def test_returns_answer_and_updated_history():
    with patch("claude_client._get_client") as mock_get_client, \
         patch("claude_client.load_pdf_text", return_value="MANUAL"):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_response("The oil capacity is 1.5L")
        mock_get_client.return_value = mock_client

        answer, history = claude_client.ask("Oil capacity?", "fake.pdf", api_history=[])

        assert answer == "The oil capacity is 1.5L"
        assert history[-1] == {"role": "assistant", "content": "The oil capacity is 1.5L"}
