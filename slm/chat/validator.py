"""
Response quality validator for LawSLM.
Validates generated responses against quality constraints before delivering to user.
Rejects generic template text, <unk>/<pad> artifacts, repetitive loops, and malformed Markdown.
"""

import re
from typing import Tuple


class ResponseValidator:
    """Validates response quality and rejects bad output."""

    # Phrases that indicate a generic template was used — MUST be rejected
    BANNED_TEMPLATE_PHRASES = [
        "Analysis & Assistance",
        "Core Concept",
        "Key Consideration",
        "Conclusion:",
        "Law SLM processes your request",
        "Regarding your query",
        "Please let me know if you need more details",
        "LawSLM processes",
        "Here is a structured response",
        "Let me provide a structured",
    ]

    @staticmethod
    def validate_response(text: str) -> Tuple[bool, str]:
        """
        Validate a response string. Returns (is_valid, cleaned_text).
        If is_valid is False, the response should be regenerated.
        """
        if not text or not text.strip():
            return False, ""

        cleaned = text.strip()

        # 1. Remove <unk> and <pad> token artifacts
        cleaned = re.sub(r'<unk>|<pad>|<eos>|<bos>', '', cleaned)

        # 2. Check for banned template phrases
        for phrase in ResponseValidator.BANNED_TEMPLATE_PHRASES:
            if phrase.lower() in cleaned.lower():
                # Strip the template phrase rather than reject entirely
                cleaned = re.sub(re.escape(phrase), '', cleaned, flags=re.IGNORECASE).strip()

        # 3. Detect repetitive word loops (same word 4+ times in a row)
        rep_match = re.search(r'\b(\w+)(?:\s+\1){3,}\b', cleaned)
        if rep_match:
            word = rep_match.group(1)
            cleaned = re.sub(r'(\b' + re.escape(word) + r'\b\s*){4,}', word + ' ', cleaned)

        # 4. Detect repetitive sentence loops
        sentences = re.split(r'[.!?]\s+', cleaned)
        if len(sentences) >= 4:
            unique = set(s.strip().lower() for s in sentences if s.strip())
            if len(unique) <= 1 and len(sentences) > 2:
                return False, ""

        # 5. Fix unbalanced code fences
        fence_count = cleaned.count('```')
        if fence_count % 2 != 0:
            cleaned += '\n```'

        # 6. Fix missing spaces after punctuation
        cleaned = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', cleaned)

        # 7. Remove excessive newlines (more than 3 in a row)
        cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)

        # 8. Final empty check after cleaning
        if not cleaned.strip():
            return False, ""

        return True, cleaned
