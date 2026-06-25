# Langchain Prompt Project

A simple Streamlit demo for sending prompts to an Ollama-backed language model using `langchain-ollama`.

## What this project does

- `prompt_ui.py` provides a small web UI where you can type a prompt and receive a text response.
- `prompt_generator.py` is intended for prompt construction or templating logic.

## Prompt flow

1. Open the Streamlit app.
2. Type a natural-language prompt into the input box.
3. Click **Submit**.
4. The app sends the prompt to the `tinyllama` model via `ChatOllama`.
5. The model response is displayed beneath the input.

## Installation

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run prompt_ui.py
```

Then open the local URL shown in the terminal.

## Notes

- The current app uses `ChatOllama(model="tinyllama")`.
- Update `prompt_generator.py` if you want to build prompt templates, prompt chaining, or prompt validation.
- This is a prompt-oriented prototype: the main entry point is `prompt_ui.py`, and prompts are sent directly to the model.

