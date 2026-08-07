"""
Multi-turn Conversation Memory & Context Management for LawSLM.
Tracks history per session, resolves coreferences ('this', 'that', 'previous answer'), and builds prompts.
"""

from typing import List, Dict, Any, Optional


class ConversationMemory:
    """
    Stores and manages conversation turns for multi-turn contextual inference.
    """

    def __init__(self, max_history_turns: int = 10) -> None:
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Adds a message turn to memory."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-self.max_history_turns * 2:]

    def clear(self) -> None:
        """Clears memory."""
        self.history.clear()

    def resolve_coreference(self, prompt: str) -> str:
        """
        Resolves references like 'this', 'that', 'previous answer', 'above' by attaching last topic context.
        """
        if not self.history:
            return prompt

        p_lower = prompt.lower()
        references = ["this", "that", "it", "previous answer", "above", "earlier", "same code", "explain that"]

        if any(r in p_lower for r in references):
            # Find last assistant message or last user message
            last_assistant_msg = next((m["content"] for m in reversed(self.history) if m["role"] == "assistant"), "")
            last_user_msg = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), "")
            
            context_summary = last_user_msg or last_assistant_msg
            if context_summary:
                # Take first 100 chars of last topic
                topic = context_summary.split("\n")[0][:100]
                return f"[Context Topic: {topic}] {prompt}"

        return prompt

    def build_prompt_context(self, current_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Formats full multi-turn conversation string with system prompt injection.
        """
        resolved_prompt = self.resolve_coreference(current_prompt)
        prompt_parts: List[str] = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for msg in self.history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role_label}: {msg['content']}\n")

        prompt_parts.append(f"User: {resolved_prompt}\nAssistant:")
        return "\n".join(prompt_parts)
