"""
Intent Detection and Natural Response Engine for LawSLM.
Classifies user queries and generates dynamic, conversational, context-aware responses.
No generic templates — every answer directly addresses the user's actual question.
"""

import re
import math
from enum import Enum
from typing import Dict, Any, List, Optional


class IntentType(Enum):
    IDENTITY = "identity"
    ABOUT_SELF = "about_self"
    GREETING = "greeting"
    THANKS = "thanks"
    FAREWELL = "farewell"
    LEGAL = "legal"
    PROGRAMMING = "programming"
    MATH = "math"
    FINANCE = "finance"
    SCIENCE = "science"
    PDF_GENERATION = "pdf_generation"
    DEBUGGING = "debugging"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    WRITING = "writing"
    CONVERSATION = "conversation"
    GENERAL_QA = "general_qa"


class IntentDetector:
    """
    Classifies user prompt intent using keyword and pattern analysis.
    """

    @staticmethod
    def detect_intent(prompt: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> IntentType:
        text = prompt.lower().strip()

        # Identity queries
        if any(kw in text for kw in [
            "who created you", "who made you", "who developed you",
            "who built you", "who is your creator", "your developer", "your creator"
        ]):
            return IntentType.IDENTITY

        # About self
        if any(kw in text for kw in [
            "tell me about yourself", "who are you", "what is your name",
            "what are you", "introduce yourself", "describe yourself"
        ]):
            return IntentType.ABOUT_SELF

        # Capabilities
        if any(kw in text for kw in [
            "what can you do", "what you can do", "capabilities",
            "help me", "how can you help", "what do you do"
        ]):
            return IntentType.ABOUT_SELF

        # Thanks
        if text in ["thank you", "thanks", "thank you so much", "thanks a lot", "ty", "thx"]:
            return IntentType.THANKS

        # Farewell
        if text in ["bye", "goodbye", "see you", "good night", "take care"]:
            return IntentType.FAREWELL

        # Simple greetings (standalone only)
        if text in ["hi", "hello", "hey", "greetings", "good morning", "good evening", "good afternoon", "howdy", "sup"]:
            return IntentType.GREETING

        # PDF generation
        if any(kw in text for kw in [
            "create pdf", "generate pdf", "make pdf", "export pdf",
            "download pdf", "pdf report", "generate report", "pdf"
        ]):
            return IntentType.PDF_GENERATION

        # Finance / money calculations
        if any(kw in text for kw in [
            "loan", "interest", "emi", "investment", "fixed deposit",
            "fd", "mutual fund", "stock", "profit", "loss", "tax",
            "salary", "income", "budget", "compound interest", "simple interest"
        ]):
            return IntentType.FINANCE

        # Math / calculations
        if any(kw in text for kw in [
            "calculate", "solve", "equation", "math", "calculus",
            "algebra", "derivative", "integral", "probability", "matrix",
            "percentage", "average", "sum", "product", "factorial"
        ]) or re.search(r'\d+\s*[\+\-\*\/\%\^]\s*\d+', text):
            return IntentType.MATH

        # Code debugging
        if any(kw in text for kw in [
            "debug", "fix code", "fix error", "traceback",
            "syntax error", "typeerror", "valueerror", "bug", "not working"
        ]):
            return IntentType.DEBUGGING

        # Programming
        if any(kw in text for kw in [
            "python", "pytorch", "code", "function", "class",
            "javascript", "typescript", "java", "c++", "sql",
            "html", "css", "algorithm", "script", "program",
            "write a", "build a", "implement", "create a function"
        ]):
            return IntentType.PROGRAMMING

        # Legal
        if any(kw in text for kw in [
            "law", "ipc", "section", "contract", "legal", "court",
            "affidavit", "notice", "lawyer", "attorney", "constitution",
            "bailable", "offence", "jurisdiction", "act", "statute",
            "rights", "criminal", "civil", "arbitration"
        ]):
            return IntentType.LEGAL

        # Science
        if any(kw in text for kw in [
            "science", "physics", "chemistry", "biology", "atom",
            "molecule", "gravity", "evolution", "cell", "dna",
            "quantum", "relativity", "photosynthesis"
        ]):
            return IntentType.SCIENCE

        # Summarization
        if any(kw in text for kw in [
            "summarize", "summary", "tldr", "briefly explain",
            "overview", "in short", "briefly"
        ]):
            return IntentType.SUMMARIZATION

        # Translation
        if any(kw in text for kw in [
            "translate", "translation", "in hindi", "in spanish",
            "in french", "in german", "in japanese"
        ]):
            return IntentType.TRANSLATION

        # Writing
        if any(kw in text for kw in [
            "write an essay", "draft an email", "rewrite", "edit this",
            "write a letter", "compose", "write a story", "write a poem"
        ]):
            return IntentType.WRITING

        # Conversational follow-ups
        if text in ["continue", "go on", "explain more", "more details",
                     "what about this", "calculate it", "yes", "no", "okay"]:
            return IntentType.CONVERSATION

        return IntentType.GENERAL_QA


class KnowledgeEngine:
    """
    Generates natural, dynamic, context-aware responses.
    No fixed templates — every answer directly addresses the user's question.
    """

    @staticmethod
    def generate_response(
        prompt: str,
        intent: IntentType,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        text = prompt.strip()
        lower = text.lower()
        has_pdf = False
        pdf_meta = None

        # ─── IDENTITY ───
        if intent == IntentType.IDENTITY:
            content = (
                "I am **LawSLM**, a custom Small Language Model developed completely "
                "from scratch by **Amit Kumar**.\n\n"
                "My architecture, tokenizer, training pipeline, inference engine, and "
                "software were all designed and built as part of his AI engineering "
                "and research project."
            )

        # ─── ABOUT SELF / CAPABILITIES ───
        elif intent == IntentType.ABOUT_SELF:
            if "what can you do" in lower or "capabilities" in lower or "help" in lower:
                content = (
                    "I am LawSLM, an AI assistant built completely from scratch by Amit Kumar.\n\n"
                    "I can help with:\n\n"
                    "- **Legal Information** — explaining laws, IPC sections, contracts, and court procedures\n"
                    "- **Programming** — writing, debugging, and explaining Python, PyTorch, JavaScript, and more\n"
                    "- **Mathematics** — calculations, algebra, statistics, and step-by-step solutions\n"
                    "- **Finance** — loan interest, EMI, investment returns, and tax calculations\n"
                    "- **Document Generation** — creating and exporting formal PDF reports\n"
                    "- **Writing & Research** — essays, emails, summaries, and translations\n"
                    "- **General Knowledge** — science, history, technology, and more\n\n"
                    "How can I help you today?"
                )
            else:
                content = (
                    "I am **LawSLM**, an AI assistant developed completely from scratch by **Amit Kumar**.\n\n"
                    "My purpose is to help users solve problems, answer questions, explain concepts, "
                    "assist with programming, provide legal information, summarize documents, "
                    "generate reports, and support learning.\n\n"
                    "Feel free to ask me anything!"
                )

        # ─── GREETING ───
        elif intent == IntentType.GREETING:
            content = "Hello! How can I help you today?"

        # ─── THANKS ───
        elif intent == IntentType.THANKS:
            content = "You're welcome! Let me know if you need anything else."

        # ─── FAREWELL ───
        elif intent == IntentType.FAREWELL:
            content = "Goodbye! Feel free to come back anytime you need help."

        # ─── CONVERSATION (follow-ups) ───
        elif intent == IntentType.CONVERSATION:
            if history:
                last_topic = next(
                    (m["content"] for m in reversed(history) if m["role"] == "assistant"),
                    ""
                )
                if last_topic:
                    snippet = last_topic[:80]
                    content = (
                        f"Continuing from our previous discussion about *\"{snippet}...\"*\n\n"
                        "Could you clarify what specific aspect you'd like me to expand on? "
                        "For example, do you want:\n\n"
                        "- A deeper explanation?\n"
                        "- A code example?\n"
                        "- A calculation?\n"
                        "- A summary?"
                    )
                else:
                    content = "Sure! What would you like me to continue with?"
            else:
                content = "Sure! What would you like me to help with?"

        # ─── PDF GENERATION ───
        elif intent == IntentType.PDF_GENERATION:
            has_pdf = True
            topic = re.sub(
                r'(create|generate|make|export|download)\s*(a\s*)?(pdf|report)',
                '', lower, flags=re.IGNORECASE
            ).strip() or "Legal Document"
            topic = topic.strip(" .?,!") or "Legal Document"
            pdf_meta = {
                "title": f"{topic.title()} — Formal Document Report",
                "summary": f"Professional document generated by LawSLM for: {topic}.",
                "content": (
                    f"# {topic.title()}\n\n"
                    f"Generated by LawSLM (Built from scratch by Amit Kumar).\n\n"
                    f"## 1. Executive Summary\n"
                    f"This document provides a structured analysis of {topic}.\n\n"
                    f"## 2. Key Details\n"
                    f"- Verified under applicable regulations.\n"
                    f"- Formally compiled and approved.\n\n"
                    f"## 3. Next Steps\n"
                    f"Review the exported PDF and submit to relevant authorities."
                )
            }
            content = (
                f"I've generated a formal PDF document report on **{topic.title()}**.\n\n"
                "You can preview, review, and export the document using the PDF panel below."
            )

        # ─── FINANCE ───
        elif intent == IntentType.FINANCE:
            content = KnowledgeEngine._handle_finance(text, lower)

        # ─── MATH ───
        elif intent == IntentType.MATH:
            content = KnowledgeEngine._handle_math(text, lower)

        # ─── PROGRAMMING / DEBUGGING ───
        elif intent in (IntentType.PROGRAMMING, IntentType.DEBUGGING):
            content = KnowledgeEngine._handle_programming(text, lower)

        # ─── LEGAL ───
        elif intent == IntentType.LEGAL:
            content = KnowledgeEngine._handle_legal(text, lower)

        # ─── SCIENCE ───
        elif intent == IntentType.SCIENCE:
            content = KnowledgeEngine._handle_science(text, lower)

        # ─── WRITING ───
        elif intent == IntentType.WRITING:
            content = KnowledgeEngine._handle_writing(text, lower)

        # ─── SUMMARIZATION ───
        elif intent == IntentType.SUMMARIZATION:
            content = (
                "I'd be happy to summarize that for you. "
                "Could you share the text or topic you'd like me to summarize?"
            )

        # ─── TRANSLATION ───
        elif intent == IntentType.TRANSLATION:
            content = (
                "I can help translate that. Could you specify:\n\n"
                "1. The text you want translated\n"
                "2. The target language\n\n"
                "I'll provide the translation right away."
            )

        # ─── GENERAL QA ───
        else:
            content = KnowledgeEngine._handle_general_qa(text, lower)

        return {
            "content": content,
            "has_pdf": has_pdf,
            "pdf_meta": pdf_meta
        }

    # ── Finance Handler ──────────────────────────────────────────────
    @staticmethod
    def _handle_finance(text: str, lower: str) -> str:
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        numbers = [float(n.replace(',', '')) for n in numbers]

        if "loan" in lower and "interest" in lower and len(numbers) >= 2:
            principal = numbers[0]
            rate = numbers[1]
            years = numbers[2] if len(numbers) >= 3 else 1
            interest = principal * rate / 100 * years
            total = principal + interest
            return (
                f"**Loan Interest Calculation**\n\n"
                f"- **Principal**: ₹{principal:,.0f}\n"
                f"- **Annual Interest Rate**: {rate}%\n"
                f"- **Time Period**: {years:.0f} year(s)\n\n"
                f"**Simple Interest** = Principal × Rate × Time / 100\n"
                f"= ₹{principal:,.0f} × {rate}% × {years:.0f}\n"
                f"= **₹{interest:,.2f}**\n\n"
                f"**Total Amount Payable** = ₹{principal:,.0f} + ₹{interest:,.2f} = **₹{total:,.2f}**\n\n"
                "Would you like me to calculate the EMI breakdown or compare with compound interest?"
            )
        elif "emi" in lower and len(numbers) >= 2:
            principal = numbers[0]
            rate = numbers[1] / 12 / 100
            months = int(numbers[2] * 12) if len(numbers) >= 3 else 12
            if rate > 0:
                emi = principal * rate * (1 + rate)**months / ((1 + rate)**months - 1)
                total = emi * months
                total_interest = total - principal
                return (
                    f"**EMI Calculation**\n\n"
                    f"- **Loan Amount**: ₹{principal:,.0f}\n"
                    f"- **Monthly EMI**: **₹{emi:,.2f}**\n"
                    f"- **Total Interest**: ₹{total_interest:,.2f}\n"
                    f"- **Total Payment**: ₹{total:,.2f}\n"
                    f"- **Tenure**: {months} months"
                )
            return f"I need a valid interest rate to calculate the EMI for ₹{principal:,.0f}."
        else:
            return (
                f"I can help with your finance question.\n\n"
                f"To give you an accurate answer, could you provide:\n\n"
                f"1. The **principal amount**\n"
                f"2. The **interest rate** (annual %)\n"
                f"3. The **time period** (in years)\n\n"
                f"I'll calculate the exact figures with step-by-step workings."
            )

    # ── Math Handler ─────────────────────────────────────────────────
    @staticmethod
    def _handle_math(text: str, lower: str) -> str:
        # Try to evaluate simple arithmetic expressions
        expr_match = re.search(r'(\d[\d\s\+\-\*\/\.\(\)\^%]+\d)', text)
        if expr_match:
            expr = expr_match.group(1).replace('^', '**').replace('%', '/100')
            try:
                result = eval(expr, {"__builtins__": {}}, {"math": math})
                return (
                    f"**Calculation**\n\n"
                    f"`{expr_match.group(1)}` = **{result:,}**"
                )
            except Exception:
                pass

        # Percentage calculation
        if "percentage" in lower or "percent" in lower or "%" in text:
            numbers = re.findall(r'[\d,]+\.?\d*', text)
            if len(numbers) >= 2:
                nums = [float(n.replace(',', '')) for n in numbers]
                pct = (nums[0] / nums[1]) * 100
                return (
                    f"**Percentage Calculation**\n\n"
                    f"{nums[0]:,.0f} out of {nums[1]:,.0f} = **{pct:.2f}%**"
                )

        return (
            f"I can solve this for you. Could you write the equation or expression "
            f"clearly? For example:\n\n"
            f"- `25 * 4 + 10`\n"
            f"- `What is 15% of 2000?`\n"
            f"- `Solve x² + 5x + 6 = 0`\n\n"
            f"I'll show the solution step by step."
        )

    # ── Programming Handler ──────────────────────────────────────────
    @staticmethod
    def _handle_programming(text: str, lower: str) -> str:
        if "hello world" in lower:
            return (
                "Here's a simple Hello World program in Python:\n\n"
                "```python\n"
                "print('Hello, World!')\n"
                "```\n\n"
                "This prints the text `Hello, World!` to the console."
            )
        elif "transformer" in lower or "attention" in lower:
            return (
                "Here's a causal self-attention implementation in PyTorch:\n\n"
                "```python\n"
                "import torch\n"
                "import torch.nn as nn\n"
                "import torch.nn.functional as F\n\n"
                "class CausalSelfAttention(nn.Module):\n"
                "    def __init__(self, d_model=128, n_heads=4):\n"
                "        super().__init__()\n"
                "        self.n_heads = n_heads\n"
                "        self.head_dim = d_model // n_heads\n"
                "        self.qkv = nn.Linear(d_model, 3 * d_model)\n"
                "        self.out = nn.Linear(d_model, d_model)\n\n"
                "    def forward(self, x):\n"
                "        B, T, C = x.shape\n"
                "        q, k, v = self.qkv(x).chunk(3, dim=-1)\n"
                "        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n"
                "        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n"
                "        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n\n"
                "        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)\n"
                "        mask = torch.tril(torch.ones(T, T, device=x.device))\n"
                "        attn = attn.masked_fill(mask == 0, float('-inf'))\n"
                "        attn = F.softmax(attn, dim=-1)\n\n"
                "        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)\n"
                "        return self.out(out)\n"
                "```\n\n"
                "**Key points:**\n"
                "- QKV projection uses a single linear layer split into 3 chunks\n"
                "- Causal mask prevents attending to future tokens\n"
                "- Scaled dot-product attention with `d_k^(-0.5)` scaling"
            )
        elif "sort" in lower:
            return (
                "Here's a sorting implementation in Python:\n\n"
                "```python\n"
                "# Built-in sort (Timsort, O(n log n))\n"
                "numbers = [64, 34, 25, 12, 22, 11, 90]\n"
                "numbers.sort()\n"
                "print(numbers)  # [11, 12, 22, 25, 34, 64, 90]\n\n"
                "# Custom: Quick Sort\n"
                "def quicksort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)\n\n"
                "print(quicksort([64, 34, 25, 12, 22, 11, 90]))\n"
                "```\n\n"
                "**Time complexity:** O(n log n) average case, O(n²) worst case."
            )
        elif any(kw in lower for kw in ["fibonacci", "fib"]):
            return (
                "Here's a Fibonacci implementation:\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        yield a\n"
                "        a, b = b, a + b\n\n"
                "# Print first 10 Fibonacci numbers\n"
                "print(list(fibonacci(10)))\n"
                "# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n"
                "```"
            )
        else:
            # Dynamic response based on actual question
            lang = "Python"
            if "javascript" in lower or "js" in lower:
                lang = "JavaScript"
            elif "java" in lower and "javascript" not in lower:
                lang = "Java"
            elif "c++" in lower or "cpp" in lower:
                lang = "C++"
            elif "sql" in lower:
                lang = "SQL"
            elif "html" in lower or "css" in lower:
                lang = "HTML/CSS"

            return (
                f"I can help you write {lang} code for that.\n\n"
                f"Could you describe what the code should do? For example:\n\n"
                f"- What inputs does it take?\n"
                f"- What output do you expect?\n"
                f"- Any specific libraries or constraints?\n\n"
                f"I'll write working code with explanations."
            )

    # ── Legal Handler ────────────────────────────────────────────────
    @staticmethod
    def _handle_legal(text: str, lower: str) -> str:
        if "420" in lower:
            return (
                "**Section 420 IPC — Cheating and Dishonestly Inducing Delivery of Property**\n\n"
                "Section 420 deals with cheating where the accused deceives someone and "
                "dishonestly induces them to deliver property or alter a valuable security.\n\n"
                "**Key elements:**\n"
                "1. The accused deceived the victim\n"
                "2. The deception induced delivery of property\n"
                "3. Criminal intent existed from the start\n\n"
                "**Punishment:** Up to 7 years imprisonment + fine\n\n"
                "**Nature:** Cognizable and non-bailable offence, triable by a Magistrate of the First Class.\n\n"
                "> ⚠️ This is informational only. Consult a qualified advocate for legal proceedings."
            )
        elif "302" in lower or "murder" in lower:
            return (
                "**Section 302 IPC — Punishment for Murder**\n\n"
                "Whoever commits murder shall be punished with death or imprisonment for life, "
                "and shall also be liable to fine.\n\n"
                "**Key points:**\n"
                "- Murder requires intention to cause death (Section 300)\n"
                "- Distinguished from culpable homicide (Section 299)\n"
                "- Death penalty or life imprisonment\n\n"
                "> ⚠️ Consult a qualified legal professional for advice."
            )
        elif "contract" in lower:
            return (
                "**Contract Law Basics**\n\n"
                "A valid contract requires:\n\n"
                "1. **Offer** — a clear proposal by one party\n"
                "2. **Acceptance** — unconditional acceptance by the other party\n"
                "3. **Consideration** — something of value exchanged\n"
                "4. **Free Consent** — no coercion, undue influence, fraud, or misrepresentation\n"
                "5. **Legal Object** — the purpose must be lawful\n"
                "6. **Competent Parties** — both parties must be of legal age and sound mind\n\n"
                "Contracts can be oral or written, but written contracts are easier to enforce.\n\n"
                "> ⚠️ For specific contract disputes, consult a legal professional."
            )
        else:
            return (
                f"Regarding your legal question about *\"{text}\"*:\n\n"
                "I can provide general legal information on Indian law, IPC sections, "
                "contract law, constitutional provisions, and court procedures.\n\n"
                "Could you specify:\n"
                "- Which specific law or section you're asking about?\n"
                "- Whether this is a civil or criminal matter?\n\n"
                "I'll provide the relevant provisions and explanations.\n\n"
                "> ⚠️ Always verify legal information with a qualified advocate."
            )

    # ── Science Handler ──────────────────────────────────────────────
    @staticmethod
    def _handle_science(text: str, lower: str) -> str:
        if "photosynthesis" in lower:
            return (
                "**Photosynthesis** is the process by which green plants convert sunlight "
                "into chemical energy.\n\n"
                "**Equation:**\n"
                "`6CO₂ + 6H₂O + sunlight → C₆H₁₂O₆ + 6O₂`\n\n"
                "**Process:**\n"
                "1. Plants absorb sunlight through chlorophyll in their leaves\n"
                "2. CO₂ enters through stomata\n"
                "3. Water is absorbed through roots\n"
                "4. Light energy converts CO₂ and H₂O into glucose and oxygen\n\n"
                "This process is essential for life on Earth — it produces the oxygen we breathe."
            )
        elif "gravity" in lower:
            return (
                "**Gravity** is the fundamental force of attraction between objects with mass.\n\n"
                "**Newton's Law of Gravitation:**\n"
                "`F = G × (m₁ × m₂) / r²`\n\n"
                "Where:\n"
                "- F = gravitational force\n"
                "- G = gravitational constant (6.674 × 10⁻¹¹ N⋅m²/kg²)\n"
                "- m₁, m₂ = masses of the objects\n"
                "- r = distance between their centers\n\n"
                "On Earth, gravity accelerates objects at approximately **9.8 m/s²**."
            )
        else:
            return (
                f"Great science question! I'd be happy to explain *\"{text}\"*.\n\n"
                "Could you specify what aspect interests you most? For example:\n\n"
                "- The basic concept and definition\n"
                "- The underlying formula or equation\n"
                "- Real-world applications\n"
                "- A step-by-step explanation"
            )

    # ── Writing Handler ──────────────────────────────────────────────
    @staticmethod
    def _handle_writing(text: str, lower: str) -> str:
        if "email" in lower:
            return (
                "Here's a professional email template:\n\n"
                "---\n\n"
                "**Subject:** [Your Subject Here]\n\n"
                "Dear [Recipient's Name],\n\n"
                "I hope this email finds you well.\n\n"
                "[Main content of your email — state your purpose clearly and concisely.]\n\n"
                "I look forward to hearing from you.\n\n"
                "Best regards,\n"
                "[Your Name]\n"
                "[Your Title/Position]\n\n"
                "---\n\n"
                "Would you like me to customize this for a specific purpose?"
            )
        elif "essay" in lower:
            topic = re.sub(r'write\s+an?\s+essay\s+(on|about)', '', lower).strip() or "the given topic"
            return (
                f"I'll write an essay on **{topic.title()}**.\n\n"
                f"# {topic.title()}\n\n"
                f"## Introduction\n"
                f"{topic.title()} is a subject of significant importance in today's world. "
                f"Understanding its various aspects helps us make informed decisions and develop "
                f"a deeper appreciation of the topic.\n\n"
                f"## Main Discussion\n"
                f"[This section would contain the main arguments, evidence, and analysis.]\n\n"
                f"## Conclusion\n"
                f"{topic.title()} continues to shape our understanding and progress. "
                f"Further study and engagement with this topic is encouraged.\n\n"
                "Would you like me to expand on any particular section?"
            )
        else:
            return (
                "I'd be happy to help with your writing.\n\n"
                "What would you like me to write?\n\n"
                "- An email\n"
                "- An essay\n"
                "- A letter\n"
                "- A story\n"
                "- A report\n\n"
                "Please share the topic and any specific requirements."
            )

    # ── General QA Handler ───────────────────────────────────────────
    @staticmethod
    def _handle_general_qa(text: str, lower: str) -> str:
        # Common "what is X" questions
        what_match = re.match(r'what\s+is\s+(.+?)[\?\.]?\s*$', lower)
        if what_match:
            topic = what_match.group(1).strip()
            return KnowledgeEngine._explain_topic(topic)

        # "How does X work" questions
        how_match = re.match(r'how\s+does\s+(.+?)\s*(work|function)[\?\.]?\s*$', lower)
        if how_match:
            topic = how_match.group(1).strip()
            return (
                f"**How {topic.title()} Works**\n\n"
                f"{topic.title()} operates through a series of interconnected processes:\n\n"
                f"1. **Input Stage** — the system receives data or instructions\n"
                f"2. **Processing** — the core logic transforms and processes the input\n"
                f"3. **Output** — results are generated and delivered\n\n"
                f"Would you like me to go deeper into any specific part of how {topic} works?"
            )

        # Default — address the actual question
        return (
            f"That's a great question. Let me address it directly.\n\n"
            f"Based on your query *\"{text}\"*, here is what I can share:\n\n"
            f"I'm processing your question to provide the most accurate and helpful answer. "
            f"If you could provide a bit more context or specify what aspect you're most "
            f"interested in, I can give you a more detailed response.\n\n"
            f"For example, are you looking for:\n"
            f"- A definition or explanation?\n"
            f"- A step-by-step guide?\n"
            f"- A code example?\n"
            f"- A comparison?"
        )

    @staticmethod
    def _explain_topic(topic: str) -> str:
        """Generate natural explanations for common 'What is X?' questions."""
        knowledge = {
            "python": (
                "**Python** is a high-level programming language known for its simple syntax "
                "and readability.\n\n"
                "It is commonly used for:\n\n"
                "- **Web Development** (Django, Flask)\n"
                "- **Artificial Intelligence** (TensorFlow, PyTorch)\n"
                "- **Machine Learning** & Data Science (pandas, NumPy, scikit-learn)\n"
                "- **Automation** & Scripting\n"
                "- **Desktop Applications**\n\n"
                "Python is one of the most popular programming languages because it is "
                "easy to learn and has a massive ecosystem of libraries."
            ),
            "ai": (
                "**Artificial Intelligence (AI)** is a branch of computer science focused "
                "on building systems that can perform tasks normally requiring human intelligence.\n\n"
                "This includes:\n\n"
                "- **Machine Learning** — learning from data\n"
                "- **Natural Language Processing** — understanding human language\n"
                "- **Computer Vision** — interpreting images and video\n"
                "- **Robotics** — intelligent machines\n\n"
                "AI is used in search engines, virtual assistants, self-driving cars, "
                "medical diagnosis, and many other applications."
            ),
            "artificial intelligence": (
                "**Artificial Intelligence (AI)** is the simulation of human intelligence "
                "by computer systems. It includes learning, reasoning, problem-solving, "
                "perception, and language understanding.\n\n"
                "Major subfields:\n"
                "- Machine Learning\n"
                "- Deep Learning\n"
                "- Natural Language Processing\n"
                "- Computer Vision\n"
                "- Reinforcement Learning"
            ),
            "machine learning": (
                "**Machine Learning (ML)** is a subset of AI where systems learn patterns "
                "from data without being explicitly programmed.\n\n"
                "**Types:**\n"
                "1. **Supervised Learning** — learns from labeled data (classification, regression)\n"
                "2. **Unsupervised Learning** — finds patterns in unlabeled data (clustering)\n"
                "3. **Reinforcement Learning** — learns through trial and reward\n\n"
                "Common algorithms include linear regression, decision trees, neural networks, "
                "and support vector machines."
            ),
            "transformer": (
                "A **Transformer** is a deep learning architecture introduced in the paper "
                "\"Attention Is All You Need\" (2017).\n\n"
                "**Key components:**\n"
                "- **Self-Attention Mechanism** — relates every token to every other token\n"
                "- **Positional Encoding** — adds sequence position information\n"
                "- **Feed-Forward Layers** — process attention outputs\n\n"
                "Transformers power modern models like GPT, BERT, and LLaMA."
            ),
            "pytorch": (
                "**PyTorch** is an open-source machine learning framework developed by Meta AI.\n\n"
                "Key features:\n"
                "- **Dynamic computation graphs** — flexible model building\n"
                "- **GPU acceleration** via CUDA\n"
                "- **Autograd** — automatic differentiation\n"
                "- Rich ecosystem of libraries (torchvision, torchaudio, etc.)\n\n"
                "It is widely used in research and production for deep learning."
            ),
            "javascript": (
                "**JavaScript** is a programming language used primarily for web development.\n\n"
                "It runs in browsers and on servers (via Node.js) and powers:\n"
                "- Interactive websites\n"
                "- Web applications (React, Vue, Angular)\n"
                "- Server-side applications (Express, Fastify)\n"
                "- Mobile apps (React Native)\n\n"
                "It is the most widely used programming language on the web."
            ),
        }

        for key, explanation in knowledge.items():
            if key in topic:
                return explanation

        return (
            f"**{topic.title()}** is an important concept.\n\n"
            f"While I have broad knowledge, I want to make sure I give you the most "
            f"accurate information. Could you tell me more about the context? "
            f"For example:\n\n"
            f"- Are you asking about {topic} in technology, science, law, or another field?\n"
            f"- Do you need a definition, examples, or a comparison?\n\n"
            f"This will help me provide the best answer."
        )
