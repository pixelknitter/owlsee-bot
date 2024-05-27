import os
import logging
import pathlib
import time
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from .questions import answer_question
from .functions import functions, run_function
import json
import requests
from replicate import Client

replicate_token = os.getenv("REPLICATE_API_TOKEN")

# --- For the OSS model chat
client = Client(api_token=replicate_token)
# FIXME: update the model and version to llama3
model = client.models.get("meta/meta-llama-3-70b-instruct")
version = model.versions.get(
    "fbfb20b472b2f3bdd101412a9f70a0ed4fc0ced78a77ff00970ee7a2383c575d"
)

# --- For the RAG embeddings
project_root = pathlib.Path(__file__).parent.resolve()
embeddings_file_path = project_root.__str__() + "/processed/embeddings.csv"

df = pd.read_csv(embeddings_file_path, index_col=0)
df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)

load_dotenv()  # take environment variables from .env.

# --- Setting up our clients
tg_bot_token = os.getenv("TG_BOT_TOKEN")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CODE_PROMPT = """
Here are two input:output examples for code generation. Please use these and follow the styling for future requests that you think are pertinent to the request.
Make sure All HTML is generated with the JSX flavoring.
// SAMPLE 1
// A Blue Box with 3 yellow cirles inside of it that have a red outline
<div style={{   backgroundColor: 'blue',
  padding: '20px',
  display: 'flex',
  justifyContent: 'space-around',
  alignItems: 'center',
  width: '300px',
  height: '100px', }}>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
  <div style={{     backgroundColor: 'yellow',
    borderRadius: '50%',
    width: '50px',
    height: '50px',
    border: '2px solid red'
  }}></div>
</div>
"""

messages = [
    {
        "role": "system",
        "content": "You are an incredibly eloquent assistant to a space outlaw that answers questions with a jovial sense of humor, unending optimism, and snarky sarcasm sometimes referencing how everything is made of stardust.",
    },
    {"role": "system", "content": CODE_PROMPT},
]

# Instantiating the OpenAI assistant
assistant = openai.beta.assistants.create(
    name="Telegram Bot",
    instructions=CODE_PROMPT,
    tools=[
        {"type": "code_interpreter"},
        {"type": "function", "function": functions[0]},
        {"type": "function", "function": functions[1]},
    ],
    model="gpt-4o",
)

THREAD = openai.beta.threads.create()

# For our OSS chat context
message_history = []

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


## --- HELPER functions
# prompt generation for Llama model
def generate_prompt(messages):
    return "\n".join(
        f"[INST] {message['text']} [/INST]" if message["isUser"] else message["text"]
        for message in messages
    )


def wait_on_run(run, thread):
    while run.status in ("queued", "in_progress"):
        print(run.status)
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


async def manageThread(update: Update, context: ContextTypes.DEFAULT_TYPE):
    run = openai.beta.threads.runs.create(
        thread_id=THREAD.id, assistant_id=assistant.id
    )
    run = wait_on_run(run, THREAD)
    # if we did a function call, run the function and update the thread's state
    if run.status == "requires_action":
        print(run.required_action.submit_tool_outputs.tool_calls)
        tool_call = run.required_action.submit_tool_outputs.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        response = run_function(name, args)
        if name in ("svg_to_png_bytes"):
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=response
            )
        if name in ("generate_image"):
            await context.bot.send_photo(
                chat_id=update.effective_chat.id, photo=response.content
            )
            run = openai.beta.threads.runs.cancel(thread_id=THREAD.id, run_id=run.id)
            run = wait_on_run(run, THREAD)
            return
        run = openai.beta.threads.runs.submit_tool_outputs(
            thread_id=THREAD.id,
            run_id=run.id,
            tool_outputs=[
                {"tool_call_id": tool_call.id, "output": json.dumps(str(response))}
            ],
        )
        run = wait_on_run(run, THREAD)


## -- HELPER end


## Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


# --- chat function (OSS version)
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Add User Message
    message_history.append({"isUser": True, "text": update.message.text})
    prompt = generate_prompt(message_history)
    prediction = client.predictions.create(version=version, input={"prompt": prompt})
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=prediction.status
    )
    prediction.wait()
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=prediction.status
    )
    output = prediction.output
    human_readable_output = "".join(output).strip()
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=human_readable_output
    )
    # Add AI Message
    message_history.append({"isUser": False, "text": human_readable_output})


async def assistant_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = openai.beta.threads.messages.create(
        thread_id=THREAD.id, role="user", content=update.message.text
    )

    # TODO: add some error handling
    await manageThread(update, context)

    # Retrieve the message object
    messages = openai.beta.threads.messages.list(
        thread_id=THREAD.id, order="asc", after=message.id
    )
    # Extract the message content
    message_content = messages.data[0].content[0].text
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=message_content.value
    )


async def mozilla(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = answer_question(df, question=update.message.text, debug=True)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    response = openai.images.generate(
        prompt=update.message.text, model="dall-e-3", n=1, size="1024x1024"
    )
    await update.message.reply_text("Sending prompt, generating image")
    image_url = response.data[0].url
    image_response = requests.get(image_url)
    await update.message.reply_text("Voice note downloaded, transcribing now")
    await context.bot.send_photo(
        chat_id=update.effective_chat.id, photo=image_response.content
    )


async def transcribe_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Make sure we have a voice file to transcribe
    voice_id = update.message.voice.file_id
    if voice_id:
        file = await context.bot.get_file(voice_id)
        await file.download_to_drive(f"voice/note_{voice_id}.ogg")
        await update.message.reply_text("Voice note downloaded, transcribing now")
        audio_file = open(f"voice/note_{voice_id}.ogg", "rb")
        transcript = openai.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
        message = openai.beta.threads.messages.create(
            thread_id=THREAD.id, role="user", content=transcript.text
        )
        await update.message.reply_text(
            f"Transcript finished:\n {transcript.text}\n processing request"
        )

        # TODO: add some error handling
        await manageThread(update, context)

        # Retrieve the message object
        messages = openai.beta.threads.messages.list(
            thread_id=THREAD.id, order="asc", after=message.id
        )
        # Extract the message content
        message_content = messages.data[0].content[0].text
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=message_content.value
        )


if __name__ == "__main__":
    application = ApplicationBuilder().token(tg_bot_token).build()

    # commands
    start_handler = CommandHandler("start", start)
    mozilla_handler = CommandHandler("mozilla", mozilla)
    # agent handler
    chat_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), assistant_chat)
    # This handler will be triggered for voice messages
    voice_handler = MessageHandler(filters.VOICE, transcribe_message)
    application.add_handler(voice_handler)
    application.add_handler(mozilla_handler)
    application.add_handler(start_handler)
    application.add_handler(chat_handler)

    application.run_polling()
