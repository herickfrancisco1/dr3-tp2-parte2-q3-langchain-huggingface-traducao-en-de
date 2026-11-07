"""API de tradução EN -> DE: FastAPI + LangChain + HuggingFace (Helsinki-NLP/opus-mt-en-de).

Execução:  uvicorn app.main:app --reload
Docs:      http://127.0.0.1:8000/docs
"""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app.chain import build_hf_llm, build_translation_chain
from app.config import get_settings
from app.schemas import HealthResponse, TranslationRequest, TranslationResponse
from app.translator import LangChainTranslator, TranslatorProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega o modelo uma vez e monta a cadeia LangChain."""
    settings = get_settings()
    logger.info("Carregando '%s' via LangChain HuggingFacePipeline...", settings.hf_model)
    llm = build_hf_llm(settings)
    tokenizer = llm.pipeline.tokenizer
    app.state.translator = LangChainTranslator(
        chain=build_translation_chain(llm),
        count_tokens=lambda text: len(tokenizer.encode(text)),
        model_name=settings.hf_model,
        max_chunk_tokens=settings.max_chunk_tokens,
    )
    logger.info("Modelo carregado.")
    yield


app = FastAPI(
    title="EN -> DE Translation API (LangChain + HuggingFace)",
    description="Traduz inglês -> alemão com Helsinki-NLP/opus-mt-en-de orquestrado pelo LangChain.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_translator(request: Request) -> TranslatorProtocol:
    translator = getattr(request.app.state, "translator", None)
    if translator is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modelo ainda não carregado.")
    return translator


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=get_settings().hf_model,
        model_loaded=getattr(request.app.state, "translator", None) is not None,
    )


# Rota `def`: a inferência local é CPU-bound; o FastAPI a executa em thread pool.
@app.post("/translate", response_model=TranslationResponse, tags=["tradução"])
def translate(
    payload: TranslationRequest,
    translator: TranslatorProtocol = Depends(get_translator),
) -> TranslationResponse:
    start = time.perf_counter()
    try:
        result = translator.translate(payload.text)
    except Exception as exc:
        logger.exception("Falha na tradução")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Erro ao traduzir o texto.") from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Tradução de %d trecho(s) em %.0f ms", result.chunks, elapsed_ms)
    return TranslationResponse(
        original_text=payload.text,
        translated_text=result.text,
        model=translator.model_name,
        chunks=result.chunks,
        elapsed_ms=round(elapsed_ms, 2),
    )
