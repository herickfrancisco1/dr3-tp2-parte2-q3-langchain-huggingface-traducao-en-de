"""Serviço de tradução: divide textos longos, chama a cadeia LangChain em lote e remonta o texto."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.runnables import Runnable

from app.text_chunking import chunk_text


@dataclass(frozen=True)
class TranslationResult:
    text: str
    chunks: int


class TranslatorProtocol(Protocol):
    model_name: str

    def translate(self, text: str) -> TranslationResult: ...


class LangChainTranslator:
    def __init__(
        self,
        chain: Runnable,
        count_tokens: Callable[[str], int],
        model_name: str,
        max_chunk_tokens: int = 400,
    ) -> None:
        self._chain = chain
        self._count_tokens = count_tokens
        self._max_chunk_tokens = max_chunk_tokens
        self.model_name = model_name

    def translate(self, text: str) -> TranslationResult:
        paragraphs = chunk_text(text, self._count_tokens, self._max_chunk_tokens)
        flat_chunks = [chunk for paragraph in paragraphs for chunk in paragraph]

        # `batch` envia todos os trechos de uma vez; o HuggingFacePipeline os agrupa em lotes
        # de `batch_size` para o modelo.
        outputs = self._chain.batch([{"text": chunk} for chunk in flat_chunks])
        translations = iter(o.strip() for o in outputs)

        translated_paragraphs = [" ".join(next(translations) for _ in p) for p in paragraphs]
        return TranslationResult(text="\n\n".join(translated_paragraphs), chunks=len(flat_chunks))
