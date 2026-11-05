"""Local LLM inference generator using Qwen/Qwen3-0.6B.

Implements lazy loading and hardware-aware execution to minimize latency.
"""

import os
import re
from typing import Any

SYSTEM_PROMPT = (
    "You are a precise technical assistant for the vLLM codebase.\n"
    "Answer the user's question concisely, accurately, and strictly grounded "
    "in the provided context.\n"
    "Do not extrapolate or hallucinate. If the context does not contain "
    "sufficient information to answer the question, state:\n"
    '"The provided context does not contain sufficient information to '
    'answer this question."'
)


class LLMGenerator:
    """Lazy-loaded local LLM generator for grounded technical answers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_new_tokens: int = 256,
    ) -> None:
        """Initialize the LLMGenerator without loading weights.

        Args:
            model_name: Hugging Face model repository or local path.
            max_new_tokens: Maximum tokens to generate per answer.
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def load_model(self) -> None:
        """Load tokenizer and model on demand with optimal dtype and device."""
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        target_dtype = torch.bfloat16 if not has_gpu else "auto"

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=target_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        except (ValueError, OSError, RuntimeError, TypeError):
            device_target = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            fallback_model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )
            self._model = fallback_model.to(device_target)

        self._model.eval()

    def get_tokenizer(self) -> Any:
        """Return the loaded tokenizer, initializing it if needed."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def format_prompt(self, question: str, context: str) -> str:
        """Format question and context into the model chat template.

        Args:
            question: User question string.
            context: Formatted context blocks from context builder.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        tokenizer = self.get_tokenizer()
        clean_context = context.strip() if context else "No context available."
        user_content = (
            f"Context:\n{clean_context}\n\nQuestion:\n{question.strip()}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    def generate_answer(self, question: str, context: str) -> str:
        """Generate a grounded answer for a question using retrieved context.

        Args:
            question: User question text.
            context: Context string containing source snippets.

        Returns:
            Generated concise answer text.
        """
        self.load_model()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        prompt = self.format_prompt(question=question, context=context)
        raw_inputs = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self._model, "device", None)
        inputs: Any
        if hasattr(raw_inputs, "to"):
            try:
                inputs = raw_inputs.to(device)
            except (TypeError, ValueError, AttributeError):
                inputs = raw_inputs
        elif isinstance(raw_inputs, dict):
            inputs = {}
            for k, v in raw_inputs.items():
                if hasattr(v, "to") and device is not None:
                    try:
                        inputs[k] = v.to(device)
                    except (TypeError, ValueError, AttributeError):
                        inputs[k] = v
                else:
                    inputs[k] = v
        else:
            inputs = raw_inputs

        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][prompt_len:]
        raw_answer: str = self._tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True,
        )

        return re.sub(
            r"<think>.*?</think>",
            "",
            raw_answer,
            flags=re.DOTALL,
        ).strip()
