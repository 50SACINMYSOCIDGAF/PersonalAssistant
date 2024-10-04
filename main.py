from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import requests
from typing import List
from sentence_transformers import SentenceTransformer, util
import numpy as np
from embedding_storage import EmbeddingStorage

app = FastAPI()

# Configure LLM server
LLM_SERVER_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL_ID = "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf"

# Initialize Stella model
stella_model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True).cuda()
query_prompt_name = "s2p_query"

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

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()

@app.get("/remember")
async def remember(query: str = Query(...)):
    try:
        query_embedding = stella_model.encode([query], prompt_name=query_prompt_name)[0]
        stored_embeddings = embedding_storage.get_all_embeddings()
        stored_texts = embedding_storage.get_all_texts()

        if stored_embeddings:
            similarities = util.pytorch_cos_sim(query_embedding, np.array(list(stored_embeddings.values())))[0]
            most_similar_idx = similarities.argmax().item()
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/teach")
async def teach(request: Request):
    try:
        form = await request.form()
        text = form["text"]
        embedding = stella_model.encode([text])[0]
        embedding_storage.add_embedding(text, embedding)
        return {"response": f"New information stored: {text}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)