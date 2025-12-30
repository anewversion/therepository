# AIgent (IamAIgo)

A local “AI teammate” project: a chat-style desktop app that can **reason**, **use tools**, and **take real actions** on your machine—like web research, creating a `.txt` file, and emailing results—while keeping the workflow in *your* hands.

This repo folder is **Project 2 of 3** inside `therepository`.

---

## What it does (high level)

- **Desktop chat UI** (Tkinter)
- **Tool-style actions**, including:
  - Web search (DuckDuckGo)
  - Local file creation / updates
  - Email sending via SMTP (attachments supported)
  - Memory via local SQLite (`memory/iam_aigo_memory.db`)
- **Model routing** (OpenAI + optional local model via Ollama)

Screenshots:
- UI + tool-call output: `docs/images/IamAIgo_Program_and_Chat_Screenshot.jpg`
- Email sent confirmation: `docs/images/AI_Agent_Email_Sent.jpg`
- Example generated text file: `docs/images/AI_Agent_Text_File_Created_and_Emailed.jpg`

---

## Quickstart

### 1) Install dependencies
From this folder:

```bash
pip install -r requirements.txt
```

### 2) Create your `.env`
Copy `ExampleEnv.txt` to a file named `.env` and replace the asterisks:

- `OPENAI_API_KEY`
- `EMAIL_USER`
- `EMAIL_PASSWORD`

Optional overrides (the app has defaults):
- `SMTP_SERVER` (default: `smtp.gmail.com`)
- `SMTP_PORT` (default: `587`)
- `OPENAI_MODEL` (default: `gpt-4o`)
- `AIGENT_TEMPERATURE` (default: `0.7`)
- `OLLAMA_MODEL` (default: `llama3.2`)

> Gmail note: you may need an **App Password** (recommended) rather than your normal login password.

### 3) Run
```bash
python IamAIgo_Ultimate.py
```

---

## Notes about paths (important)
This version was built and tested in a Windows environment. Some file paths are currently set to a Windows user directory (e.g. `C:\Users\...`).
If you run on a different machine/username/OS, update the path settings inside `IamAIgo_Ultimate.py` to match your environment.

---

## Safety / expectations

- This is a powerful pattern: a model + tools + real actions. Run it with care.
- Do **not** commit your real `.env` file or credentials.
- The included SQLite DB (`memory/iam_aigo_memory.db`) is part of the demo setup for this folder.

---

## License
MIT (see repo root).
