# Astro-AI-Kotha
Voice-enabled support / FAQ interface in English, Bangla and Banglish

### Development installation

For local development:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```
