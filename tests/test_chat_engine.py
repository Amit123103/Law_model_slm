"""
Unit tests for LawSLM Intent Detector, Conversation Memory, and Response Validator modules.
"""

import pytest
from slm.chat.intent import IntentDetector, KnowledgeEngine, IntentType
from slm.chat.memory import ConversationMemory
from slm.chat.validator import ResponseValidator


def test_intent_detector_identity():
    intent = IntentDetector.detect_intent("Who created you?")
    assert intent == IntentType.IDENTITY


def test_intent_detector_pdf():
    intent = IntentDetector.detect_intent("Create a PDF explaining Python")
    assert intent == IntentType.PDF_GENERATION


def test_intent_detector_legal():
    intent = IntentDetector.detect_intent("Explain Section 420 of IPC")
    assert intent == IntentType.LEGAL


def test_intent_detector_programming():
    intent = IntentDetector.detect_intent("Write a Python script for Transformer")
    assert intent == IntentType.PROGRAMMING


def test_knowledge_engine_response():
    resp = KnowledgeEngine.generate_response("Who created you?", IntentType.IDENTITY)
    assert "Amit Kumar" in resp["content"]
    assert "LawSLM" in resp["content"]


def test_conversation_memory():
    mem = ConversationMemory()
    mem.add_message("user", "What is Python?")
    mem.add_message("assistant", "Python is a programming language.")
    
    resolved = mem.resolve_coreference("Explain that in detail")
    assert "[Context Topic: What is Python?]" in resolved


def test_response_validator():
    is_valid, cleaned = ResponseValidator.validate_response("hi", "Hello <unk> world <pad>")
    assert is_valid is True
    assert "<unk>" not in cleaned
    assert "<pad>" not in cleaned
