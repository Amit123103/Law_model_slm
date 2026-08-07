# REST API Documentation - Small Language Model (SLM)

## Launching REST API Server

Start the FastAPI application using Uvicorn:

```bash
uvicorn slm.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is automatically served at: `http://localhost:8000/docs`.

## Endpoints Summary

### 1. GET `/health`
Returns service status.

### 2. GET `/info`
Returns model parameter count, layer configuration, and active device.

### 3. POST `/generate`
Autoregressive text generation.

**Request Body:**
```json
{
  "prompt": "Deep learning architectures",
  "max_new_tokens": 64,
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "repetition_penalty": 1.1
}
```

### 4. POST `/tokenizer/encode`
Encodes text to token IDs.

### 5. POST `/tokenizer/decode`
Decodes token IDs back to text.
