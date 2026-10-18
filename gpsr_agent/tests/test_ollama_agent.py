from unittest.mock import patch

import pytest
from langchain_ollama import ChatOllama
from ollama import ResponseError

from gpsr_agent import ollama_agent


def test_default_model_uses_ollama_defaults(monkeypatch):
    monkeypatch.delenv("GPSR_OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("GPSR_OLLAMA_BASE_URL", raising=False)
    monkeypatch.setattr(ollama_agent, "load_dotenv", lambda: None)
    with patch.object(ollama_agent, "build_shared_agent") as shared:
        ollama_agent.build_agent()
        model = shared.call_args.kwargs["model"]
        assert isinstance(model, ChatOllama)
        assert model.model == "llama3.1:latest"
        assert model.base_url is None


def test_model_and_url_override(monkeypatch):
    monkeypatch.setenv("GPSR_OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("GPSR_OLLAMA_BASE_URL", "http://localhost:11435")
    monkeypatch.setattr(ollama_agent, "load_dotenv", lambda: None)
    with patch.object(ollama_agent, "build_shared_agent") as shared:
        ollama_agent.build_agent()
        model = shared.call_args.kwargs["model"]
        assert model.model == "qwen3:8b"
        assert model.base_url == "http://localhost:11435"


def test_unsupported_tools_error():
    with patch.object(ollama_agent, "build_agent"):
        with patch.object(ollama_agent, "run_cli", side_effect=ResponseError(
            "model does not support tools", status_code=400
        )):
            with pytest.raises(SystemExit, match="GPSR_OLLAMA_MODEL=llama3.1:latest"):
                ollama_agent.main()
