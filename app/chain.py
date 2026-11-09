"""Integração LangChain <-> HuggingFace: o modelo MarianMT local é encapsulado como um LLM do
LangChain (`HuggingFacePipeline`) e composto em uma cadeia LCEL."""

from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from app.config import Settings

# Modelos de tradução MarianMT NÃO seguem instruções: qualquer texto no prompt seria traduzido junto.
# Por isso o template apenas repassa o texto ("{text}").
TRANSLATION_PROMPT = PromptTemplate.from_template("{text}")


def build_hf_llm(settings: Settings):
    """Baixa (1ª vez) e carrega o modelo localmente como um LLM do LangChain."""
    from langchain_huggingface import HuggingFacePipeline

    return HuggingFacePipeline.from_model_id(
        model_id=settings.hf_model,
        task="translation",
        device=settings.device,
        batch_size=settings.batch_size,
        pipeline_kwargs={"max_length": 512},
    )


def build_translation_chain(llm: BaseLLM) -> Runnable:
    """Monta a cadeia LCEL: prompt de repasse -> HuggingFacePipeline -> parser de texto."""
    return TRANSLATION_PROMPT | llm | StrOutputParser()
