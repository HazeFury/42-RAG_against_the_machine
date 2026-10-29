"""Unit tests for LLMGenerator in src/generation/llm_generator.py."""

from unittest.mock import MagicMock

from src.generation.llm_generator import LLMGenerator, SYSTEM_PROMPT


def test_generator_lazy_initialization() -> None:
    """Verify LLMGenerator does not load weights at initialization time."""
    gen = LLMGenerator(model_name="Qwen/Qwen3-0.6B")
    assert gen._model is None
    assert gen._tokenizer is None
    assert gen.model_name == "Qwen/Qwen3-0.6B"


def test_generator_format_prompt_mocked() -> None:
    """Test chat prompt construction with mocked tokenizer."""
    gen = LLMGenerator(model_name="mock-model")

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = (
        "<system>prompt</system><user>Context:\nctx\n\nQuestion:\nq</user>"
    )
    gen._tokenizer = mock_tokenizer

    prompt = gen.format_prompt(question="q", context="ctx")

    assert "Context:\nctx" in prompt
    assert "Question:\nq" in prompt
    mock_tokenizer.apply_chat_template.assert_called_once()
    call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert call_args[0]["role"] == "system"
    assert call_args[0]["content"] == SYSTEM_PROMPT
    assert call_args[1]["role"] == "user"


def test_generator_generate_answer_mocked() -> None:
    """Test generation pipeline and thinking tag stripping with mocks."""
    gen = LLMGenerator(model_name="mock-model")

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"
    # Simulate tokenizer encoding into dummy tensor
    import torch

    dummy_input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.return_value = {"input_ids": dummy_input_ids}
    mock_tokenizer.decode.return_value = (
        "<think>internal thoughts</think>Final concise answer."
    )

    mock_model = MagicMock()
    # Output includes prompt + generated tokens
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.to.return_value = mock_model

    gen._tokenizer = mock_tokenizer
    gen._model = mock_model

    ans = gen.generate_answer(question="What is X?", context="Context about X")
    assert ans == "Final concise answer."
    assert "<think>" not in ans
