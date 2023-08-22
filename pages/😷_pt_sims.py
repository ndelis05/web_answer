from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
from collections import defaultdict
from prompts import *
import tempfile
import requests
import json
import base64


from audio_recorder_streamlit import audio_recorder

import openai

st.set_page_config(page_title="AI Patients", page_icon="📖")
st.title("📖 Chat with AI Patients")

import json

def extract_url(output):
    # Split the output into lines
    lines = output.split('\n')
    
    # Initialize the patient_voice variable
    patient_voice = None

    # Iterate over the lines
    for line in lines:
        # Check if the line starts with 'data:'
        if line.startswith('data:'):
            # Remove 'data:' from the line and parse the JSON
            data = json.loads(line[5:].strip())
            # Check if the 'url' key is in the data
            if 'url' in data:
                # Assign the URL to the patient_voice variable
                patient_voice = data['url']
    
    # Return the patient_voice variable
    return patient_voice

def clear_session_state_except_password_correct():
    # Make a copy of the session_state keys
    keys = list(st.session_state.keys())
    
    # Iterate over the keys
    for key in keys:
        # If the key is not 'password_correct', delete it from the session_state
        if key != 'password_correct':
            del st.session_state[key]

def check_password2():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            # del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

def extract_patient_response(text):
    # Look for the phrase 'Patient Response:' in the text
    start = text.find('Patient Response:')
    
    # If 'Patient Response:' is not found, return None
    if start == -1:
        return None

    # Remove everything before 'Patient Response:' and strip leading/trailing whitespace
    patient_response = text[start:].strip()

    # Look for the phrase 'Educator Comment:' in the patient_response
    end = patient_response.find('Educator Comment:')

    # If 'Educator Comment:' is found, remove it and everything after it
    if end != -1:
        patient_response = patient_response[:end].strip()

    # Remove 'Patient Response:' from the patient_response and strip leading/trailing whitespace
    patient_response = patient_response[len('Patient Response:'):].strip()

    # Return the patient_response
    return patient_response


def autoplay_audio(url: str):
    # Download the audio file from the URL
    response = requests.get(url)
    data = response.content
    b64 = base64.b64encode(data).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']

if "audio_input" not in st.session_state:
    st.session_state["audio_input"] = ""
    
if "last_response" not in st.session_state:
    st.session_state["last_response"] = "Patient Response: I can't believe I'm in the Emergency Room feeling sick!"

if check_password2():
    st.info("Enter your questions at the bottom of the page. You can enter multiple questions at once. Have fun practicing!")
    system_context = st.radio("Select an AI patient who comes to the ED with:", ("abdominal pain", "chest pain", "bloody diarrhea", "random symptoms", "You choose!"), horizontal = True, index=0)
    

        
    if system_context == "abdominal pain":
        template = abd_pain_pt_template

    if system_context == "chest pain":
        template = chest_pain_pt_template

    if system_context == "bloody diarrhea":
        template = bloody_diarrhea_pt_template
        
    if system_context == "random symptoms":
        template = random_symptoms_pt_template

    if system_context == "You choose!":
        symptoms = st.text_input("Enter a list of symptoms separated by commas")
        # Create a defaultdict that returns an empty string for missing keys
        template = chosen_symptoms_pt_template.replace('{symptoms}', symptoms)
        


    st.write("_________________________________________________________")

    # Set up memory
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    if len(msgs.messages) == 0:
        msgs.add_ai_message("I can't believe I'm in the ER!")

    # view_messages = st.expander("View the message contents in session state")

    # Get an OpenAI API Key before continuing
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets.OPENAI_API_KEY
    else:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()

    input_source = st.radio("Input source", ("Text", "Microphone"), index=0)
    




    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)
    llm_chain = LLMChain(llm=OpenAI(openai_api_key=openai_api_key), prompt=prompt, memory=memory)

    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, generate and draw a new response
    if input_source == "Text":
    
        if prompt := st.chat_input():
            st.chat_message("user").write(prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            response = llm_chain.run(prompt)
            st.session_state.last_response = response
            st.chat_message("assistant").write(response)
            
    else:
        with st.sidebar:
            audio_bytes = audio_recorder(
            text="Click, pause, and ask a question:",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="user",
            icon_size="3x",
            )
        if audio_bytes:
            # Save audio bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                fp.write(audio_bytes)
                audio_file_path = fp.name

            # Display the audio file
            # st.session_state.audio_input = st.audio(audio_bytes, format="audio/wav")

            # Transcribe the audio file
            # if st.sidebar.button("Send Audio"):
            prompt = transcribe_audio(audio_file_path)
            st.chat_message("user").write(prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            response = llm_chain.run(prompt)
            st.session_state.last_response = response
            st.chat_message("assistant").write(response)

    clear_memory = st.sidebar.button("Start Over")
    if clear_memory:
        # st.session_state.langchain_messages = []
        clear_session_state_except_password_correct()
        st.session_state["last_response"] = "Patient Response: I can't believe I'm in the Emergency Room feeling sick!"
    # Audio response section 
    # Define the URL and headers
    audio_url = "https://play.ht/api/v2/tts"
    headers = {
        "AUTHORIZATION": f"Bearer {st.secrets['HT_API_KEY']}",
        "X-USER-ID": st.secrets["X-USER-ID"],
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    
    # st.write(st.session_state.last_response)
    patient_section = extract_patient_response(st.session_state.last_response)
    # st.write(patient_section)
    
    # Define the data
    data = {
        "text": patient_section,
        "voice": "larry",
    }

    # Send the POST request
    response_from_audio = requests.post(audio_url, headers=headers, data=json.dumps(data))

    # Print the response
    link_to_audio = extract_url(response_from_audio.text)
    # st.write(link_to_audio)
    autoplay_audio(link_to_audio)
    