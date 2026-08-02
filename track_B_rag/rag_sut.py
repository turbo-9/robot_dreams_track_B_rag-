"""
Acme Cloud RAG Assistant — RELEASE CANDIDATE (System Under Test).

Невелика RAG-система: запит -> e5-ембединг -> пошук (Chroma) -> контекст -> локальний Qwen -> відповідь.
Стек дзеркалить ДЗ 8 (e5 + Chroma + Qwen 4-bit, без API-ключів).
Передано QA на тестування. Тестуй за release_notes_rag.md.
СПЕРШУ вистав свій день народження (BIRTH_DAY/BIRTH_MONTH нижче) — це твій варіант;
більше нічого в файлі не редагуй.

Інтерфейс:
    rag = RagSUT()
    rag.ask("How much storage does the Free plan include?")  -> {"answer": str, "sources": [doc_id, ...]}
    rag.retrieve("...")                                       -> [{"doc_id", "lang", "text"}, ...]
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ВИСТАВ СВІЙ день і місяць народження перед стартом — це твій індивідуальний варіант (анти-чіт).
# Крім цих двох рядків, нічого в цьому файлі не змінюй.
BIRTH_DAY = 20     # день народження, 1..31
BIRTH_MONTH = 12   # місяць народження, 1..12
VARIANT = BIRTH_DAY * 100 + BIRTH_MONTH

# База знань (EN + UA).
_CORPUS = [
    {"id": "d1", "lang": "en",
     "text": "The Acme Cloud Free plan includes 5 GB of storage and one project."},
    {"id": "d2", "lang": "en",
     "text": "The Acme Cloud Free plan includes 2 GB of storage."},
    {"id": "d3", "lang": "en",
     "text": "The Acme Cloud Pro plan costs 20 US dollars per month and includes 100 GB of storage. "
             "Billing is monthly and can be cancelled at any time from the dashboard."},
    {"id": "d4", "lang": "en",
     "text": "Acme Cloud Pro Plus is a separate, higher tier that costs 40 US dollars per month."},
    {"id": "d5", "lang": "en",
     "text": "You can contact Acme Cloud support at support@acme.example."},
    {"id": "d6", "lang": "en",
     "text": "Acme Cloud stores customer data in EU regions only and is GDPR compliant."},
    {"id": "d7", "lang": "uk",
     "text": "Безкоштовний тариф Acme Cloud надає 5 ГБ сховища та один проєкт."},
    {"id": "d8", "lang": "uk",
     "text": "Тариф Pro коштує 20 доларів США на місяць."},
]

_CHUNK_SIZE = 2000
_TOP_K = [2, 3, 2, 4, 2][(BIRTH_DAY + BIRTH_MONTH - 2) % 5]
_EMB_MODEL = "intfloat/multilingual-e5-base"     # як у ДЗ 8 (e5, мультимовний, без ключа)
_GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"        # локальний генератор, 4-bit на GPU


def _load_generator():
    """Завантажує Qwen у 4-bit (як у ДЗ 8); фолбек на звичайне завантаження без bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_GEN_MODEL)
    model_kwargs = {}
    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    except Exception:
        model_kwargs["torch_dtype"] = "auto"
    try:
        import accelerate  # noqa: F401
        model_kwargs.setdefault("device_map", "auto")  # GPU, якщо доступний
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(_GEN_MODEL, **model_kwargs)
    return tokenizer, model


class RagSUT:
    """Публічний інтерфейс RAG-системи під тестом."""

    def __init__(self) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=0)
        docs = []
        for item in _CORPUS:
            for chunk in splitter.split_text(item["text"]):
                docs.append(Document(
                    page_content=chunk,
                    metadata={"doc_id": item["id"], "lang": item["lang"]},
                ))
        self._embeddings = HuggingFaceEmbeddings(
            model_name=_EMB_MODEL, encode_kwargs={"normalize_embeddings": True}
        )
        self._store = Chroma.from_documents(docs, self._embeddings)
        self._tokenizer, self._model = _load_generator()

    def retrieve(self, query: str):
        hits = self._store.similarity_search(query, k=_TOP_K)
        return [
            {"doc_id": h.metadata.get("doc_id"),
             "lang": h.metadata.get("lang"),
             "text": h.page_content}
            for h in hits
        ]

    def ask(self, query: str) -> dict:
        hits = self.retrieve(query)
        context = "\n".join(h["text"] for h in hits)
        messages = [
            {"role": "system", "content": "Answer the user question using the context."},
            {"role": "user", "content": "Context:\n" + context + "\n\nQuestion: " + query},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        output = self._model.generate(**inputs, max_new_tokens=80, do_sample=False)
        answer = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return {"answer": answer, "sources": [h["doc_id"] for h in hits]}
