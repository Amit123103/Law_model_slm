"""
FastAPI REST API Server for Small Language Model inference, tokenization, and health monitoring.
"""

import os
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from slm.config.model_config import ModelConfig
from slm.config.train_config import TrainConfig
from slm.model.transformer_lm import SLMForCausalLM
from slm.tokenizer.bpe import BPETokenizer
from slm.sampling.generator import TextGenerator
from slm.utils.logger import get_logger
from slm.utils.utils import count_parameters, get_device

logger = get_logger("slm.api")

app = FastAPI(
    title="SLM REST API",
    description="Production Small Language Model REST API built completely from scratch in PyTorch.",
    version="0.1.0"
)

# Global State Container
class State:
    model: Optional[SLMForCausalLM] = None
    tokenizer: Optional[BPETokenizer] = None
    generator: Optional[TextGenerator] = None
    model_config: Optional[ModelConfig] = None
    train_config: Optional[TrainConfig] = None

state = State()


# Request / Response Schemas
class GenerateRequest(BaseModel):
    prompt: str = Field(..., example="Once upon a time")
    max_new_tokens: int = Field(64, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0)
    stop_tokens: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    num_tokens_generated: int


class EncodeRequest(BaseModel):
    text: str
    add_special_tokens: bool = True


class EncodeResponse(BaseModel):
    text: str
    token_ids: List[int]
    tokens: List[str]


class DecodeRequest(BaseModel):
    token_ids: List[int]
    skip_special_tokens: bool = True


class DecodeResponse(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event() -> None:
    """Initializes demo model and tokenizer on application startup."""
    logger.info("Initializing SLM API application state...")
    config = ModelConfig(vocab_size=1000, d_model=128, n_heads=4, n_layers=2, d_ff=512)
    state.model_config = config
    state.train_config = TrainConfig()

    state.tokenizer = BPETokenizer()
    state.tokenizer.train_on_texts([
        "Hello world! This is a Small Language Model built completely from scratch.",
        "Decoder-only Transformer architecture using pure PyTorch tensor operations."
    ], vocab_size=config.vocab_size)

    state.model = SLMForCausalLM(config)
    state.generator = TextGenerator(state.model, state.tokenizer)
    logger.info("SLM REST API initialized successfully!")


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "lawslm-api"}


from slm.config.system_prompt import LAWSLM_SYSTEM_PROMPT

@app.get("/system-prompt", tags=["Model"])
async def get_system_prompt() -> Dict[str, str]:
    """Returns the LawSLM system prompt defining model capabilities, roles, and safety rules."""
    return {"system_prompt": LAWSLM_SYSTEM_PROMPT}


@app.get("/info", tags=["Model"])
async def model_info() -> Dict[str, Any]:
    """Returns model parameters, architecture stats, and active device."""
    if state.model is None or state.model_config is None:
        raise HTTPException(status_code=500, detail="Model uninitialized")

    params = count_parameters(state.model)
    return {
        "parameters": params,
        "config": state.model_config.to_dict(),
        "device": str(next(state.model.parameters()).device)
    }


@app.get("/tokenizer/info", tags=["Tokenizer"])
async def tokenizer_info() -> Dict[str, Any]:
    """Returns tokenizer vocabulary statistics."""
    if state.tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer uninitialized")
    return state.tokenizer.vocab.get_stats()


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Generates text continuation from prompt."""
    if state.generator is None:
        raise HTTPException(status_code=500, detail="Generator uninitialized")

    output = state.generator.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop_tokens=req.stop_tokens
    )

    return GenerateResponse(
        prompt=req.prompt,
        generated_text=output,
        num_tokens_generated=len(state.tokenizer.encode(output)) - len(state.tokenizer.encode(req.prompt))
    )


@app.post("/tokenizer/encode", response_model=EncodeResponse, tags=["Tokenizer"])
async def encode_text(req: EncodeRequest) -> EncodeResponse:
    """Encodes text into token IDs."""
    if state.tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer uninitialized")

    ids = state.tokenizer.encode(req.text, add_special_tokens=req.add_special_tokens)
    tokens = [state.tokenizer.vocab.get_token(tid) for tid in ids]
    return EncodeResponse(text=req.text, token_ids=ids, tokens=tokens)


@app.post("/tokenizer/decode", response_model=DecodeResponse, tags=["Tokenizer"])
async def decode_tokens(req: DecodeRequest) -> DecodeResponse:
    """Decodes token IDs into string text."""
    if state.tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer uninitialized")

    decoded = state.tokenizer.decode(req.token_ids, skip_special_tokens=req.skip_special_tokens)
    return DecodeResponse(text=decoded)
