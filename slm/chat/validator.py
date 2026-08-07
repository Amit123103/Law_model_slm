"""
Response Validation Engine for LawSLM.
Performs quality verification on generated responses before displaying output.
"""

import re
from typing import Tuple


class ResponseValidator:
    """
    Validates model output quality, formatting, and completeness.
    """

    @staticmethod
    def validate_response(prompt: str, response: str) -> Tuple[bool, str]:
        """
        Validates response string against quality checks.

        Returns:
            Tuple of (is_valid: bool, cleaned_or_fallback_response: str)
        """
        if not response or not response.strip():
            return False, "LawSLM is ready to assist. Please ask your question or specify your request."

        text = response.strip()

        # 1. Reject responses containing unk / pad token artifacts
        if "<unk>" in text or "<pad>" in text:
            cleaned = text.replace("<unk>", "").replace("<pad>", "").strip()
            if not cleaned or len(cleaned) < 10:
                return False, "I am LawSLM, a custom Small Language Model built completely from scratch by Amit Kumar. How can I help you today?"
            text = cleaned

        # 2. Check for repetitive word loops (e.g., "word word word word")
        words = text.split()
        if len(words) >= 6:
            # Check if same word repeated 5+ times in a row
            for i in range(len(words) - 4):
                if len(set(words[i:i + 5])) == 1:
                    return False, "LawSLM generated a repetitive sequence. Please retry your query for an updated response."

        # 3. Ensure balanced code fence backticks (```)
        code_fence_count = text.count("```")
        if code_fence_count % 2 != 0:
            text += "\n```"

        # 4. Check for spacing issues (e.g., words stuck together like "whatyoucando")
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        return True, text
