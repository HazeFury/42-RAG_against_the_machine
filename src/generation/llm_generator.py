"""Local LLM inference generator using Qwen/Qwen3-0.6B.

Implements lazy loading to prevent memory and runtime overhead during
purely lexical search or indexing workflows.
"""

from typing import Any, Optional
import os
import re

SYSTEM_PROMPT = (
    "You are a technical assistant for vLLM.\n"
    "Answer the user's question based on the provided source code and "
    "documentation snippets (including file names, classes, and "
    "functions).\n"
    "Be concise, accurate, and direct.\n"
    "If the context genuinely contains no relevant information, state that "
    "the context does not contain sufficient information."
)


class LLMGenerator:
    """Lazy-loaded local LLM generator for grounded technical answers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cpu",
        max_new_tokens: int = 256,
    ) -> None:
        """Initialize the LLMGenerator without loading weights.

        Args:
            model_name: Hugging Face model repository or local path.
            device: Computing device ('cpu').
            max_new_tokens: Maximum tokens to generate per answer.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def load_model(self) -> None:
        """Load tokenizer and model on demand if not already loaded."""
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Set offline env if local cache is preferred
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        # Load tokenizer with offline preference fallback
        tokenizer: Any
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model on CPU with float32 precision
        model: Any
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                local_files_only=True,
            )
        except TypeError:
            # Fallback for transformers versions expecting torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                local_files_only=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            )

        model.to(self.device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def get_tokenizer(self) -> Any:
        """Return the tokenizer, loading it if necessary.

        Returns:
            The loaded Hugging Face tokenizer.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name
                )
        return self._tokenizer

    def format_prompt(self, question: str, context: str) -> str:
        """Format question and context into chat template.

        Args:
            question: User question string.
            context: Formatted context blocks from context builder.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        tokenizer = self.get_tokenizer()

        clean_context = context.strip() if context else "No context available."
        user_content = (
            f"Context:\n{clean_context}\n\n"
            f"Question:\n{question.strip()}"
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
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        gen_tokens = outputs[0][prompt_len:]
        raw_answer: str = self._tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True,
        )

        # Strip any stray thinking tags from output
        cleaned_answer = re.sub(
            r"<think>.*?</think>",
            "",
            raw_answer,
            flags=re.DOTALL,
        ).strip()

        return cleaned_answer
