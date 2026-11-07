"""Testa a integração com o LangChain usando um LLM falso no lugar do modelo HuggingFace."""

from langchain_core.language_models.fake import FakeListLLM

from app.chain import TRANSLATION_PROMPT, build_translation_chain
from app.translator import LangChainTranslator


def count_words(text: str) -> int:
    return len(text.split())


def test_prompt_passes_text_through_unchanged():
    # Modelos MarianMT traduziriam qualquer instrução extra, então o prompt deve ser só o texto.
    assert TRANSLATION_PROMPT.format(text="Hello world") == "Hello world"


def test_text_is_split_by_sentence_batched_and_reassembled():
    llm = FakeListLLM(responses=["Satz eins.", "Satz zwei.", "Absatz zwei."])
    translator = LangChainTranslator(
        chain=build_translation_chain(llm), count_tokens=count_words, model_name="fake", max_chunk_tokens=50
    )

    result = translator.translate("Sentence one here. Sentence two here.\n\nParagraph two.")

    assert result.chunks == 3
    assert result.text == "Satz eins. Satz zwei.\n\nAbsatz zwei."
