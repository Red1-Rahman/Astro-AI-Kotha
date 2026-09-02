# Astro-AI-Kotha

Voice-enabled support and FAQ interface for Astro-AI in English, Bangla, and Banglish.

## Development installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```
