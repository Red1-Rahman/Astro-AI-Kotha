# Astro-AI-Kotha Specification

## 1. Scope

Astro-AI-Kotha is a voice-enabled FAQ interface for Astro-AI.

Supported languages:

* English
* Bangla
* Banglish

The FAQ knowledge base is authoritative. The system does not generate unsupported answers.

## 2. Input Modes

### Text

```text
text → FAQ matching → answer
```

### English Voice

```text
audio → STT → FAQ matching → answer → TTS
```

### Bangla / Banglish Voice

```text
audio
→ STT
→ language detection
→ Bangla/Banglish → English translation
→ FAQ matching
→ English answer
→ English → Bangla translation
→ TTS
```

## 3. FAQ Matching

* Knowledge base: `data/faqs.json`
* NLP: spaCy
* Model: `en_core_web_sm`
* Preprocessing:

  * tokenization
  * lemmatization
  * stop-word removal
  * punctuation/space removal
  * lowercase normalization
* Matching: TF-IDF cosine similarity
* Default threshold: `0.4`
* Scores are normalized to `[0.0, 1.0]`.
* Below-threshold matches return the fallback response.

## 4. Contracts

Primary Pydantic models:

* `FAQMetadata`
* `FAQ`
* `FAQDatabase`
* `FAQMatch`
* `RelatedQuestion`
* `ChatResponse`

Invalid FAQ data must fail validation during loading.

## 5. Voice

Speech providers are isolated behind `speech/` interfaces.

Required operations:

* speech-to-text
* text-to-speech

Audio is processed in memory and is not persisted.

Provider credentials must never be logged.

## 6. Translation

Translation is isolated behind `translation/`.

Translation is used only for Bangla/Banglish voice flows. English FAQ matching remains the authoritative retrieval path.

BanglaBERT is experimental and is not part of the required V1 retrieval path.

## 7. API

`POST /api/chat`

Input:

```json
{
  "query": "string"
}
```

Output:

```json
{
  "answer": "string",
  "score": 0.0,
  "related_questions": []
}
```

The existing text-chat endpoint remains supported.

## 8. Error Handling

* Invalid input → client error.
* Invalid FAQ data → startup/load failure.
* Speech provider failure → explicit voice error.
* TTS failure must not discard an otherwise successful text answer.
* Provider/API errors must not be silently ignored.

## 9. Non-Goals

V1 does not include:

* generative LLM answers
* vector databases
* RAG frameworks
* persistent audio storage
* user accounts
* conversation databases
* background job systems
* multi-service deployment
