"""
Unit tests for LawSLM Chat Engine:
- IntentDetector: classifies user prompts into correct intent categories
- KnowledgeEngine: generates natural, dynamic responses (no generic templates)
- ConversationMemory: tracks multi-turn history and resolves coreferences
- ResponseValidator: validates quality and rejects template artifacts
"""

import pytest
from slm.chat.intent import IntentDetector, KnowledgeEngine, IntentType
from slm.chat.memory import ConversationMemory
from slm.chat.validator import ResponseValidator


class TestIntentDetector:
    def test_identity(self):
        assert IntentDetector.detect_intent("Who created you?") == IntentType.IDENTITY
        assert IntentDetector.detect_intent("who made you") == IntentType.IDENTITY

    def test_about_self(self):
        assert IntentDetector.detect_intent("Who are you?") == IntentType.ABOUT_SELF
        assert IntentDetector.detect_intent("Tell me about yourself") == IntentType.ABOUT_SELF
        assert IntentDetector.detect_intent("What can you do?") == IntentType.ABOUT_SELF

    def test_greeting(self):
        assert IntentDetector.detect_intent("Hello") == IntentType.GREETING
        assert IntentDetector.detect_intent("hi") == IntentType.GREETING

    def test_thanks(self):
        assert IntentDetector.detect_intent("Thank you") == IntentType.THANKS
        assert IntentDetector.detect_intent("thanks") == IntentType.THANKS

    def test_farewell(self):
        assert IntentDetector.detect_intent("goodbye") == IntentType.FAREWELL

    def test_pdf_generation(self):
        assert IntentDetector.detect_intent("Generate a PDF report") == IntentType.PDF_GENERATION

    def test_legal(self):
        assert IntentDetector.detect_intent("Explain Section 420 IPC") == IntentType.LEGAL

    def test_programming(self):
        assert IntentDetector.detect_intent("Write Python code for sorting") == IntentType.PROGRAMMING

    def test_finance(self):
        assert IntentDetector.detect_intent("Calculate loan interest for 10 lakh at 12%") == IntentType.FINANCE

    def test_math(self):
        assert IntentDetector.detect_intent("Calculate 25 * 4 + 10") == IntentType.MATH

    def test_science(self):
        assert IntentDetector.detect_intent("Explain photosynthesis") == IntentType.SCIENCE


class TestKnowledgeEngine:
    def test_identity_response_mentions_amit(self):
        result = KnowledgeEngine.generate_response("Who created you?", IntentType.IDENTITY)
        assert "Amit Kumar" in result["content"]
        assert result["has_pdf"] is False

    def test_about_self_response(self):
        result = KnowledgeEngine.generate_response("Who are you?", IntentType.ABOUT_SELF)
        assert "LawSLM" in result["content"]

    def test_greeting_response_is_natural(self):
        result = KnowledgeEngine.generate_response("Hello", IntentType.GREETING)
        assert "Hello" in result["content"]
        # Must NOT contain generic template phrases
        assert "Analysis & Assistance" not in result["content"]
        assert "Core Concept" not in result["content"]

    def test_thanks_response(self):
        result = KnowledgeEngine.generate_response("Thank you", IntentType.THANKS)
        assert "welcome" in result["content"].lower()

    def test_pdf_has_metadata(self):
        result = KnowledgeEngine.generate_response("Generate PDF report", IntentType.PDF_GENERATION)
        assert result["has_pdf"] is True
        assert result["pdf_meta"] is not None

    def test_finance_with_numbers(self):
        result = KnowledgeEngine.generate_response(
            "Calculate loan interest for 100000 at 12% for 2 years", IntentType.FINANCE
        )
        assert "₹" in result["content"] or "Interest" in result["content"]

    def test_math_calculation(self):
        result = KnowledgeEngine.generate_response("Calculate 25 * 4 + 10", IntentType.MATH)
        content = result["content"]
        assert "110" in content  # 25*4+10 = 110

    def test_legal_section_420(self):
        result = KnowledgeEngine.generate_response("Explain Section 420 IPC", IntentType.LEGAL)
        assert "420" in result["content"]
        assert "cheat" in result["content"].lower() or "Cheating" in result["content"]

    def test_python_explanation(self):
        result = KnowledgeEngine.generate_response("What is Python?", IntentType.GENERAL_QA)
        assert "Python" in result["content"]

    def test_no_generic_template_in_any_response(self):
        """Verify no response uses banned template phrases."""
        test_prompts = [
            ("Hello", IntentType.GREETING),
            ("Who are you?", IntentType.ABOUT_SELF),
            ("Explain gravity", IntentType.SCIENCE),
            ("Write Python code", IntentType.PROGRAMMING),
        ]
        banned = ["Analysis & Assistance", "Core Concept", "Key Consideration"]
        for prompt, intent in test_prompts:
            result = KnowledgeEngine.generate_response(prompt, intent)
            for phrase in banned:
                assert phrase not in result["content"], (
                    f"Banned phrase '{phrase}' found in response to '{prompt}'"
                )


class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi! How can I help?")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_coreference_resolution(self):
        mem = ConversationMemory()
        mem.add_message("user", "What is Python?")
        mem.add_message("assistant", "Python is a programming language.")
        resolved = mem.resolve_coreference("Tell me more about this")
        assert "python" in resolved.lower() or "this" in resolved.lower()


class TestResponseValidator:
    def test_valid_text_passes(self):
        is_valid, cleaned = ResponseValidator.validate_response("Hello! I am LawSLM.")
        assert is_valid is True
        assert "LawSLM" in cleaned

    def test_empty_text_fails(self):
        is_valid, _ = ResponseValidator.validate_response("")
        assert is_valid is False

    def test_unk_tokens_removed(self):
        is_valid, cleaned = ResponseValidator.validate_response("Hello <unk> world <pad>")
        assert is_valid is True
        assert "<unk>" not in cleaned
        assert "<pad>" not in cleaned

    def test_repetitive_loop_cleaned(self):
        is_valid, cleaned = ResponseValidator.validate_response("the the the the the answer")
        assert is_valid is True
        assert cleaned.count("the") < 4

    def test_template_phrases_stripped(self):
        is_valid, cleaned = ResponseValidator.validate_response(
            "Analysis & Assistance\n\nHello, I am LawSLM."
        )
        assert is_valid is True
        assert "Analysis & Assistance" not in cleaned

    def test_unbalanced_code_fence_fixed(self):
        is_valid, cleaned = ResponseValidator.validate_response("```python\nprint('hello')")
        assert is_valid is True
        assert cleaned.count("```") % 2 == 0
