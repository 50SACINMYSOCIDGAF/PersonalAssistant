from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from embedding_storage import EmbeddingStorage
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure LLM server
LLM_SERVER_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_ID = "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf"

# Initialize BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Initialize EmbeddingStorage
embedding_storage = EmbeddingStorage()


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionResponse(BaseModel):
    response: str


def query_llm(messages: List[Message], temperature: float, max_tokens: int) -> str:
    response = requests.post(LLM_SERVER_URL, json={
        "model": MODEL_ID,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens
    })

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise HTTPException(status_code=500, detail="Error communicating with LLM server")


def get_embedding(text: str) -> np.ndarray:
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    except Exception as e:
        logger.error(f"Error in get_embedding: {str(e)}")
        raise


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()


@app.get("/remember")
async def remember(query: str = Query(...)):
    try:
        query_embedding = get_embedding(query)
        stored_embeddings = embedding_storage.get_all_embeddings()
        stored_texts = embedding_storage.get_all_texts()

        if stored_embeddings:
            similarities = [cosine_similarity(query_embedding, np.array(emb)) for emb in stored_embeddings.values()]
            most_similar_idx = np.argmax(similarities)
            most_similar_text = stored_texts[most_similar_idx]

            messages = [
                Message(role="system", content="""You are a friendly and knowledgeable assistant.
                Your responses should be direct, specific, and based solely on the provided context.
                Don't add extra details or make assumptions. If the context doesn't fully answer the query,
                provide only the relevant information you have. Start your responses with 'From my knowledge,'
                and focus on answering the specific question asked without elaborating on tangential information
                but while still keeping a friendly tone. You may add extra information at the end relating to
                the question but base it off your knowledge and keep it related to the question asked."""),
                Message(role="user", content=f"Context: {most_similar_text}\n\nQuery: {query}")
            ]
            response = query_llm(messages, temperature=0.2, max_tokens=50)
        else:
            response = "I don't have any information about that yet."

        return {"response": response}
    except Exception as e:
        logger.error(f"Error in remember endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/teach")
async def teach(request: Request):
    try:
        form = await request.form()
        text = form.get("text")
        if not text:
            raise ValueError("No text provided in the form data")

        logger.info(f"Received text: {text}")

        embedding = get_embedding(text)
        logger.info(f"Generated embedding shape: {embedding.shape}")

        embedding_storage.add_embedding(text, embedding)
        logger.info("Embedding added to storage")

        return {"response": f"New information stored: {text}"}
    except Exception as e:
        logger.error(f"Error in teach endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)