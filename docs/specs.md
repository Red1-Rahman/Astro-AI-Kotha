# Astro-AI-Kotha Specification

## 1. Scope

Astro-AI-Kotha is a voice-enabled FAQ interface for Astro-AI.

Supported languages:

* English
* Bangla
* Banglish

The FAQ knowledge base is authoritative. The system does not generate unsupported answers.

English is the canonical language of the FAQ knowledge base and retrieval pipeline.

## 2. Input Modes

### Text

```text
text
→ language detection
→ language-specific sanitizer
→ FAQ matching
→ answer
```

The current text FAQ path operates against the English FAQ knowledge base.

### English Voice

```text
audio
→ STT
→ language detection
→ English sanitizer
→ FAQ matching
→ answer
→ TTS
```

### Bangla / Banglish Voice

```text
audio
→ STT
→ language detection
→ language-specific sanitizer
→ Bangla/Banglish → English translation
→ English sanitizer
→ FAQ matching
→ English answer
→ English → Bangla translation
→ TTS
```

Translation is required only when the voice input or output requires a language other than English.

## 3. Language Handling

The system dynamically selects the sanitizer based on detected language.

Supported sanitizers:

* `english_sanitizer.py`
* `bangla_sanitizer.py`
* `banglish_sanitizer.py`

Language detection and sanitizer selection are separate responsibilities.

Sanitization removes language-specific conversational noise without removing domain-relevant terms.

Supported language classifications:

* English
* Bangla
* Banglish
* Unknown

Unsupported or indeterminate language input must result in an explicit error rather than an invented language classification.

## 4. FAQ Matching

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

The matcher operates against the English FAQ representation.

Bangla and Banglish queries are translated to English before entering the English FAQ matching path.

The matcher does not generate new answers.

## 5. Contracts

Primary Pydantic models:

* `FAQMetadata`
* `FAQ`
* `FAQDatabase`
* `FAQMatch`
* `RelatedQuestion`
* `ChatResponse`

Invalid FAQ data must fail validation during loading.

Translation provider contracts are defined separately from the FAQ response contracts.

The translation layer provides:

* `TranslationDirection`
* `TranslationResult`
* `Translator`
* `TranslationError`

Provider-specific failures are represented by translation-layer error types.

## 6. Voice

Speech providers are isolated behind `speech/`.

Required operations:

* speech-to-text
* text-to-speech

Audio is processed in memory and is not persisted.

Provider credentials must never be logged.

Speech operations must not block the asyncio event loop.

TTS failure must not discard an otherwise successful text answer.

## 7. Translation

Translation is isolated behind `translation/`.

Translation is used for Bangla/Banglish voice flows.

English FAQ matching remains the authoritative retrieval path.

### 7.1 Translation Providers

The default translation provider is local.

Local translation uses:

* `Helsinki-NLP/opus-mt-bn-en` for Bangla → English
* `Helsinki-NLP/opus-mt-en-bn` for English → Bangla

The local models are loaded lazily and are not downloaded or initialized during module import.

### 7.2 Azure Fallback

Azure Translator is an optional fallback provider.

The default provider flow is:

```text
Local translation
       │
       ├── success → use result
       │
       └── TranslationError
                    │
                    ▼
             Azure translation
```

Azure fallback can be disabled through configuration.

Fallback is triggered by translation failures, not merely because the application suspects that a translation may be low quality.

### 7.3 Banglish

Banglish is supported as best-effort translation input.

The local Bengali → English model is not specifically trained for Banglish. Therefore, Banglish translation may be less reliable than standard Bangla translation.

Azure fallback provides an additional recovery path when the local translation operation fails.

### 7.4 Configuration

Translation configuration is provided through environment variables:

```dotenv
TRANSLATION_PROVIDER=local
TRANSLATION_FALLBACK_ENABLED=true

AZURE_TRANSLATOR_KEY=
AZURE_TRANSLATOR_REGION=
AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
```

`TRANSLATION_PROVIDER` supports:

* `local`
* `azure`

`TRANSLATION_FALLBACK_ENABLED` controls whether Azure is used after a local translation failure.

Azure credentials are optional when Azure is not configured as the active provider or fallback provider.

Credentials must never be hardcoded, committed, or logged.

### 7.5 Async Execution

Local model inference is executed outside the asyncio event loop.

Model initialization is protected by a thread lock so concurrent requests cannot initialize the same model multiple times.

Azure translation uses asynchronous HTTP requests.

## 8. API

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

## 9. Error Handling

* Invalid input → client error.
* Invalid FAQ data → startup/load failure.
* Unsupported language → explicit error.
* Local translation failure → `LocalTranslationError`.
* Azure translation failure → `AzureTranslationError`.
* Translation failure → explicit error when no fallback succeeds.
* Speech provider failure → explicit error.
* TTS failure must not discard an otherwise successful text answer.
* Provider/API errors must not be silently ignored.
* Programming errors must not be silently converted into provider fallback requests.

## 10. Non-Goals

V1 does not include:

* generative LLM answers
* vector databases
* RAG frameworks
* persistent audio storage
* user accounts
* conversation databases
* background job systems
* multi-service deployment
* Google Cloud Translation
* LLM-based translation
