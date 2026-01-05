import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_api.main import query_llm
from src.rag_api.modules.prompt_builder import build_prompt
from src.rag_api.modules.retrieval import get_top_k_chunks
from src.rag_api.modules.translator import translate_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str
    language: str = "pl"


@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    query = request.query
    lang = request.language
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Received query: {query} | Target lang: {lang}")

    processing_query = query
    if lang != "pl":
        processing_query = translate_text(query, target_lang_code="pl")
        logger.info(f"Translated query to PL: '{processing_query}'")

    sorted_chunks = get_top_k_chunks(processing_query)

    if not sorted_chunks:
        polish_msg = "Przepraszam, nie znalazłem w bazie informacji na ten temat."
        final_msg = translate_text(polish_msg, lang) if lang != "pl" else polish_msg
        return {
            "answer": final_msg,
            "sources": [],
        }

    text_only_chunks = [chunk["text_chunk"] for chunk in sorted_chunks]
    prompt = build_prompt(processing_query, text_only_chunks)

    polish_answer = query_llm(prompt)
    final_answer = polish_answer
    if lang != "pl":
        logger.info(f"Translating answer from PL to {lang}...")
        final_answer = translate_text(polish_answer, target_lang_code=lang)

    sources = [chunk.get("source_url", "Unknown") for chunk in sorted_chunks[:5]]

    return {"answer": final_answer, "sources": sources}
