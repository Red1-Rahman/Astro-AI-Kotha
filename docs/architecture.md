# Astro-AI-Kotha Architecture

## 1. Overview

```text
Frontend
   │
   ▼
FastAPI
   │
   ├── Text ─────────────────────────┐
   │                                 ▼
   │                         Language Detector
   │                                 │
   │                                 ▼
   │                          Sanitizer Router
   │                                 │
   │                                 ▼
   │                           FAQ Matcher
   │                                 │
   │                                 ▼
   │                          Response Builder
   │
   └── Voice → Speech ───────────────┘
```

The application is a single Python service.

## 2. Components

```text
main.py
├── chatbot/
│   ├── faq_loader.py
│   ├── sanitizer/
│   │   ├── english_sanitizer.py
│   │   ├── bangla_sanitizer.py
│   │   ├── banglish_sanitizer.py
│   │   └── router.py
│   ├── nlp_processor.py
│   ├── matcher.py
│   └── response_builder.py
├── speech/
│   ├── transcriber.py
│   └── synthesizer.py
├── translation/
│   └── translator.py
├── data/
│   └── faqs.json
└── frontend/
```

## 3. Responsibilities

### `main.py`

* Creates application dependencies.
* Defines HTTP endpoints.
* Coordinates text and voice flows.
* Contains no FAQ matching or sanitization logic.

### `chatbot/`

Owns the deterministic FAQ system.

* `faq_loader.py` — loads and validates the knowledge base.
* `sanitizer/` — language detection, routing, and language-specific query cleanup.
* `nlp_processor.py` — English linguistic preprocessing.
* `matcher.py` — TF-IDF indexing and similarity matching.
* `response_builder.py` — converts matches into API responses.

### `speech/`

Owns speech provider integration.

* `transcriber.py` — audio → text.
* `synthesizer.py` — text → audio.

Provider-specific implementation details remain inside this package.

### `translation/`

Owns language translation.

```text
Bangla/Banglish ↔ English
```

It does not perform FAQ retrieval or sanitization.

## 4. Request Flows

### Text

```text
text
→ language detection
→ sanitizer router
→ language-specific sanitizer
→ translation if required
→ English NLP
→ FAQ matcher
→ response builder
```

### English Voice

```text
audio
→ transcriber
→ language detection
→ English sanitizer
→ matcher
→ response builder
→ synthesizer
→ audio
```

### Bangla/Banglish Voice

```text
audio
→ transcriber
→ language detection
→ Bangla/Banglish sanitizer
→ translation
→ English sanitizer
→ English NLP
→ FAQ matcher
→ response builder
→ translation
→ synthesizer
→ audio
```

## 5. Dependency Boundaries

```text
main
 ├── chatbot
 ├── speech
 └── translation

chatbot ── standalone
speech ── standalone
translation ── standalone
```

Within `chatbot/`:

```text
language detector
      ↓
sanitizer router
      ↓
language sanitizer
      ↓
nlp_processor
      ↓
matcher
      ↓
response_builder
```

`chatbot/` must not depend on `speech/` or `translation/`.

Provider-specific types must not cross into the FAQ domain.

## 6. Data Flow

`data/faqs.json` is the authoritative FAQ source.

```text
faqs.json
   ↓
FAQDatabase
   ↓
FAQMatcher
   ↓
FAQMatch
   ↓
ChatResponse
```

No component may invent FAQ content.

## 7. Runtime Constraints

* Async provider operations must not block the event loop.
* Audio is not persisted.
* Secrets are loaded from configuration/environment.
* Provider failures are surfaced explicitly.
* FAQ validation occurs before the matcher is used.
* External providers are mocked in unit tests.
