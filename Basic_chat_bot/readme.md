# Basic Chatbot

A minimal command-line chatbot using `langchain-ollama` and the `phi3:mini` Ollama model.

## What it does

- Starts a simple text loop in the terminal.
- Sends user input to the model as a conversation history.
- Prints the model response and continues until you type `exit`.

## Dependencies

The chatbot uses:

- `langchain-ollama`
- `langchain-core`
- `ollama`

Install dependencies from `requirements.txt` if available.

## Run the chatbot

```powershell
python chatbot.py
```

## Usage

- Type a message and press Enter.
- The bot replies with a short answer.
- Type `exit` to stop the program.

## Notes

- The conversation begins with a system prompt that instructs the assistant to stay helpful and keep responses short.
- The bot sends the entire chat history to the model on each turn.

