# Astro-AI-Kotha Architecture

```text
Frontend
   │
   ▼
FastAPI
   │
   ├── Text ───────────────────────────────────────┐
   │                                               │
   │                                               ▼
   │                                      Language Detector
   │                                               │
   │                                               ▼
   │                                        Sanitizer Router
   │                                               │
   │                                               ▼
   │                                         FAQ Matcher
   │                                               │
   │                                               ▼
   │                                       Response Builder
   │
   └── Voice
         │
         ▼
       Speech
         │
         ├── STT
         │    │
         │    ▼
         │  Language Detector
         │    │
         │    ▼
         │  Sanitizer Router
         │    │
         │    ▼
         │  Translation
         │    │
         │    ├── Local Translator
         │    │      │
         │    │      └── Helsinki-NLP models
         │    │
         │    └── Azure Translator fallback
         │
         │    ▼
         │  FAQ Matcher
         │    │
         │    ▼
         │  Response Builder
         │    │
         │    ▼
         │  Translation
         │    │
         │    ▼
         │  TTS
         │
         └─────────────────────────────────────────┘
```

## 1. Architectural Boundaries

Astro-AI-Kotha is intentionally structured as a small set of independent components.

```text
chatbot/
speech/
translation/
```

Responsibilities are separated as follows:

* `chatbot/` owns language detection, query sanitization, FAQ loading, NLP processing, FAQ matching, and response construction.
* `speech/` owns speech-to-text and text-to-speech provider integrations.
* `translation/` owns translation contracts, local translation, Azure translation, and translation fallback.
* `data/` contains the authoritative FAQ knowledge base.
* `frontend/` provides the user interface.
* `main.py` coordinates the application and API layer.

The chatbot domain must not depend directly on speech providers.

The chatbot domain must not depend directly on provider-specific translation implementations.

Provider-specific types must not cross the domain boundaries unnecessarily.

## 2. Request Flow

### Text

Text requests enter the chatbot pipeline directly:

```text
text
→ language detection
→ language-specific sanitization
→ FAQ matching
→ response builder
```

The existing FAQ matcher operates against the English FAQ knowledge base.

### English Voice

English voice requests follow:

```text
audio
→ speech-to-text
→ language detection
→ English sanitization
→ FAQ matching
→ response builder
→ text-to-speech
```

No translation is required for the English path.

### Bangla / Banglish Voice

Bangla and Banglish voice requests follow:

```text
audio
→ speech-to-text
→ language detection
→ language-specific sanitization
→ local translation
→ Azure fallback if local translation fails
→ English sanitization
→ FAQ matching
→ English answer
→ English → Bangla translation
→ text-to-speech
```

The English FAQ knowledge base remains the authoritative retrieval source.

## 3. Chatbot Layer

The chatbot layer contains:

```text
chatbot/
├── faq_loader.py
├── language_detector.py
├── sanitizer/
├── nlp_processor.py
├── matcher.py
└── response_builder.py
```

### Language Detection

`language_detector.py` identifies:

* English
* Bangla
* Banglish
* Unknown

Language detection and sanitization are separate responsibilities.

### Sanitization

The sanitizer router selects the language-specific sanitizer:

```text
Language
   │
   ├── English  → english_sanitizer.py
   ├── Bangla   → bangla_sanitizer.py
   └── Banglish → banglish_sanitizer.py
```

Sanitization removes conversational noise while preserving domain-relevant information.

### FAQ Matching

The matcher is deterministic and English-centric.

```text
query
→ spaCy processing
→ TF-IDF representation
→ cosine similarity
→ best FAQ
→ threshold check
```

The matcher does not generate answers.

## 4. Translation Layer

Translation is isolated behind the `translation/` package:

```text
translation/
├── __init__.py
├── translator.py
├── local_translator.py
└── azure_translator.py
```

The provider-independent contract is defined in `translator.py`.

The application interacts with the `Translator` abstraction rather than depending on a concrete translation provider.

### Primary Provider

Local translation is the default provider.

The local implementation uses:

* `Helsinki-NLP/opus-mt-bn-en` for Bangla → English
* `Helsinki-NLP/opus-mt-en-bn` for English → Bangla

Models are loaded lazily.

The models are not loaded during module import.

### Translation Fallback

The runtime translation path is:

```text
LocalTranslator
      │
      ├── success ───────────────→ result
      │
      └── TranslationError
                  │
                  ▼
          AzureTranslator
                  │
                  ▼
               result
```

Azure Translator is therefore an operational fallback rather than the primary translation mechanism.

Fallback is configurable and can be disabled.

### Banglish

Banglish is not treated as a separate trained machine-translation language.

Banglish text is handled as best-effort input to the Bengali → English local translation path.

Because the local Bengali model is not specifically trained for Banglish, Azure fallback provides an additional recovery path when local translation fails.

Translation quality problems that do not raise an error are not automatically detectable by the translation layer.

## 5. Async and Concurrency Model

External and potentially blocking operations must not block the FastAPI event loop.

Local translation model inference is executed through a worker thread:

```text
async translate()
      │
      ▼
asyncio.to_thread(...)
      │
      ▼
synchronous model inference
```

Local model initialization is protected by `threading.Lock`.

The lock is used only for model initialization so concurrent requests cannot load the same model multiple times.

Already-loaded models do not require the initialization lock for every translation request.

Azure translation uses asynchronous `httpx` operations.

## 6. Speech Layer

Speech providers are isolated behind `speech/`.

The speech layer provides:

* speech-to-text
* text-to-speech

The current voice provider integration uses Fish through the Vercel AI Gateway.

Audio is processed in memory.

Audio is not persisted by the application.

Speech provider credentials must never be logged or committed.

## 7. Data Flow and Knowledge Authority

The FAQ knowledge base is stored in:

```text
data/faqs.json
```

It is the authoritative source of supported answers.

Translation does not create knowledge.

The system therefore follows:

```text
User query
    │
    ▼
Language processing
    │
    ▼
Translation when required
    │
    ▼
English FAQ matching
    │
    ▼
Authoritative FAQ answer
    │
    ▼
Translation when required
    │
    ▼
User-facing response
```

No generative model is used to invent an answer when the FAQ knowledge base does not contain a sufficiently similar question.

## 8. Configuration and Secrets

Provider configuration is supplied through environment variables.

Translation configuration:

```dotenv
TRANSLATION_PROVIDER=local
TRANSLATION_FALLBACK_ENABLED=true

AZURE_TRANSLATOR_KEY=
AZURE_TRANSLATOR_REGION=
AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
```

Secrets must not be hardcoded or committed to the repository.

The application must not log API keys or other provider credentials.

## 9. Error Boundaries

Each integration exposes explicit domain-level errors.

Examples include:

```text
LocalTranslationError
AzureTranslationError
```

Provider-specific failures are converted into the appropriate translation-layer errors.

Fallback occurs only for translation failures represented by `TranslationError`.

Unexpected programming errors must not be silently converted into fallback requests.

A successful text answer must remain usable even when TTS fails.

## 10. Dependency Direction

The intended dependency direction is:

```text
Frontend
   │
   ▼
FastAPI / Application
   │
   ├── chatbot
   │
   ├── speech
   │
   └── translation
          │
          ├── local_translator
          └── azure_translator
```

The FAQ domain remains independent of concrete speech and translation providers.

Provider-specific implementations depend on their own external libraries and the provider-neutral contracts.

The translation contract does not depend on either the local Hugging Face implementation or Azure implementation.
