import io
import os
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dataclasses import dataclass
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END, START
from groq import Groq
from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

@dataclass(kw_only=True)
class Configuration:
    """
    configurable fields for the chatbot
    """
    user_id: str = "default-user"
    todo_category: str = "general"
    taskist_role: str = "You are a helpful task management assistant. You help create, organize, and manage the user's ToDo list."

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """
        create a Configuration instance from a RunnableConfig
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in cls.__dataclass_fields__.values()
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# in-memory todo list storage
todo_lists = {}  # dictionary to store todos: {user_id: {category: [tasks]}}

def record_audio_until_stop(state: MessagesState, config: RunnableConfig = None):
    """
    records audio from the microphone until 'Enter' is pressed, then transcribes it using Whisper
    """
    audio_data = []  # list to store audio chunks
    recording = True  # flag to control recording
    sample_rate = 16000  # (kHz)

    def record_audio():
        """
        continuously records audio until the recording flag is set to False
        """
        nonlocal audio_data, recording
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16') as stream:
            print("====TASKIST====")
            print("Recording your instruction! ... Press Enter to stop recording.")
            while recording:
                audio_chunk, _ = stream.read(1024)  # read audio data in chunks
                audio_data.append(audio_chunk)

    def stop_recording():
        """
        waits for user input to stop the recording
        """
        input()  # wait for Enter key press
        nonlocal recording
        recording = False

    # start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

    # start a thread to listen for the Enter key
    stop_thread = threading.Thread(target=stop_recording)
    stop_thread.start()

    # wait for both threads to complete
    stop_thread.join()
    recording_thread.join()

    # stack all audio chunks into a single numpy array and write to file
    audio_data = np.concatenate(audio_data, axis=0)

    # convert to .wav format 
    audio_bytes = io.BytesIO()
    write(audio_bytes, sample_rate, audio_data)  # scipy's write function to save to BytesIO
    audio_bytes.seek(0)  # start of the BytesIO buffer
    audio_bytes.name = "audio.wav"  

    # transcribe
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes.read()),
            model="whisper-large-v3-turbo",  
            response_format="text",  
            language="en" 
        )
        
        print("Here is the transcription:", transcription)
        return {"messages": [HumanMessage(content=transcription)]}
    except Exception as e:
        print(f"Error during transcription: {e}")
        return {"messages": [HumanMessage(content="Transcription failed. Please try again.")]}

def todo_app(state: MessagesState, config: RunnableConfig = None):
    """
    manages a todo list based on the transcribed input and configuration
    """
    # get config
    configuration = Configuration.from_runnable_config(config)

    # initialize todo list for user and category if not exists
    user_id = configuration.user_id
    category = configuration.todo_category
    if user_id not in todo_lists:
        todo_lists[user_id] = {}
    if category not in todo_lists[user_id]:
        todo_lists[user_id][category] = []

    input_message = state["messages"][-1].content.lower().strip()

    # simple command parsing
    if input_message.startswith("add "):
        task = input_message[4:].strip()
        todo_lists[user_id][category].append(task)
        response = f"Added task '{task}' to {category} todo list for user {user_id}."
    elif input_message.startswith("list"):
        tasks = todo_lists[user_id][category]
        if tasks:
            response = f"Tasks in {category} for {user_id}:\n" + "\n".join(f"- {task}" for task in tasks)
        else:
            response = f"No tasks in {category} for {user_id}."
    elif input_message.startswith("remove "):
        task = input_message[7:].strip()
        if task in todo_lists[user_id][category]:
            todo_lists[user_id][category].remove(task)
            response = f"Removed task '{task}' from {category} todo list for user {user_id}."
        else:
            response = f"Task '{task}' not found in {category} for {user_id}."
    else:
        response = f"Unknown command: {input_message}. Use 'add <task>', 'list', or 'remove <task>'."

    # print assistant role for context
    print(f"Assistant role: {configuration.taskist_role}")
    print(f"Processing todo for user: {user_id}, category: {category}")
    print(response)

    return {"messages": [HumanMessage(content=response)]}

def play_audio(state: MessagesState, config: RunnableConfig = None):
    """
    plays the audio response from the todo app
    """
    response = state['messages'][-1]

    cleaned_text = response.content.replace("**", "")

    response = elevenlabs_client.text_to_speech.convert(
        voice_id="Xb7hH8MSUJpSbSDYk0k2",
        output_format="mp3_22050_32",
        text=cleaned_text,
        model_id="eleven_flash_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    play(response)

    return {"messages": state["messages"]}

# parent graph
builder = StateGraph(MessagesState)

# add nodes with configuration support
builder.add_node("audio_input", record_audio_until_stop)
builder.add_node("todo_app", todo_app)
builder.add_node("audio_output", play_audio)

# edges
builder.add_edge(START, "audio_input")
builder.add_edge("audio_input", "todo_app")
builder.add_edge("todo_app", "audio_output")
builder.add_edge("audio_output", END)

# compile the graph
graph = builder.compile()

config = {"configurable": {"user_id": "ss", "todo_category": "personal"}}
while True:
    graph.invoke({"messages": []}, config=config)
    if input("Continue? (y/n): ").lower() != 'y':
        break