from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from audio_recorder_streamlit import audio_recorder
import streamlit as st
from collections import defaultdict
from prompts import *
import tempfile
import requests
import json
import base64
import openai
import os
import re
from elevenlabs import clone, generate, play, set_api_key, stream



st.set_page_config(page_title="Interview Practice!", page_icon="🧐")
st.title("🧐 Interview Practice")

def extract_url(text):
    # Use regular expressions to find the URL pattern
    pattern = r"url\":\"(.*?)\""
    match = re.search(pattern, text)
    
    if match:
        # Extract the URL from the matched pattern
        url = match.group(1)
        return url
    else:
        st.write("Error generating audio... Try again in a moment")
        return None

def play_audio_eleven(text, voice="Rachel"):
    set_api_key(st.secrets["ELEVEN_API_KEY"])    

    audio = generate(text=text, voice=voice, stream = False)
    filename = "pt_latest.mp3"
    with open(filename, "wb") as f:
        f.write(audio)  # write the bytes directly to the file

    # st.audio(filename, format='audio/mp3', start_time=0)

    return filename

def play_audio_eleven_all(text, voice="Rachel"):
    set_api_key(st.secrets["ELEVEN_API_KEY"])    

    audio = generate(text=text, voice=voice, stream = False)
    
    

    play(audio, notebook=False, use_ffmpeg=False)
    
    filename = "pt_latest.mp3"
    with open(filename, "wb") as f:
        f.write(audio)  # write the bytes directly to the file

    return filename

def autoplay_local_audio(filepath: str):
    # Read the audio file from the local file system
    with open(filepath, 'rb') as f:
        data = f.read()
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
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']

if "audio_off" not in st.session_state:
    st.session_state["audio_off"] = False

if "audio_input" not in st.session_state:
    st.session_state["audio_input"] = ""
    
if "last_response" not in st.session_state:
    st.session_state["last_response"] = "Hi, I'm Dr. Smith! Nice to meet you!"

if check_password2():
    st.info("Have fun. Enter responses at the bottom of the page or choose the Microphone option. This tool uses openai's GPT3.5 turbo 16k model.")
    system_context = st.radio("Select an interviewer type :", ("Tough", "Nice",), horizontal = True, index=0)
    specialty = st.text_input("Enter your specialty", placeholder="e.g. Emergency Medicine")
    position = st.text_input("Enter your position", placeholder="e.g. resident")
    
    

        
    if system_context == "Tough":
        template = tough_interviewer
        voice = 'Rachel'

    if system_context == "Nice":
        template = nice_interviewer
        voice = 'Dave'

    # if system_context == "bloody diarrhea":
    #     template = bloody_diarrhea_pt_template
    #     voice = 'david'
        
    # if system_context == "random symptoms":
    #     template = random_symptoms_pt_template
    #     voice = 'oliver'

    if system_context == "You choose!":
        symptoms = st.text_input("Enter a list of symptoms separated by commas", placeholder="e.g. fever, cough, headache after returning from a trip to Africa")
        # Create a defaultdict that returns an empty string for missing keys
        template = f'Here are the symptoms: {symptoms} and respond according to the following template:' + chosen_symptoms_pt_template
        voice = 'russell'
        
    if st.button("Set a Scenario"):
        clear_session_state_except_password_correct()
        st.session_state["last_response"] = "Hi, I'm Dr. Smith! Nice to meet you!"
    
    
    if specialty is not None and position is not None:
        formatted_template = template.format(specialty=specialty, position=position, history = "{history}", human_input = "{human_input}")
        # st.write(f'{formatted_template}')    


    st.write("_________________________________________________________")

    # Set up memory
    msgs_interview = StreamlitChatMessageHistory(key="langchain_messages_interview")
    memory = ConversationBufferMemory(chat_memory=msgs_interview)
    if len(msgs_interview.messages) == 0:
        msgs_interview.add_ai_message("Hi, I'm Dr. Smith! Nice to meet you!")

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
    st.session_state.audio_off = st.checkbox("Turn off voice generation", value=False) 




    prompt = PromptTemplate(input_variables=["history", "human_input"], template=formatted_template)
    llm_chain = LLMChain(llm=ChatOpenAI(openai_api_key=openai_api_key, model = "gpt-3.5-turbo-16k"), prompt=prompt, memory=memory)

    # Render current messages from StreamlitChatMessageHistory
    for msg in msgs_interview.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, generate and draw a new response
    if input_source == "Text":
    
        if prompt := st.chat_input():
            st.chat_message("user").write(prompt)
            # Note: new messages are saved to history automatically by Langchain during run
            openai.api_base = "https://api.openai.com/v1"
            openai.api_key = st.secrets["OPENAI_API_KEY"]
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
        st.session_state["last_response"] = "Hi, I'm Dr. Smith! Nice to meet you!"
        
    # if create_hpi := st.sidebar.button("Create HPI (Wait until you have enough history.)"):    
    #     openai.api_base = "https://api.openai.com/v1"
    #     openai.api_key = st.secrets["OPENAI_API_KEY"]    
    #     hpi = llm_chain.run(hpi_prompt)
    #     st.sidebar.write(hpi)
            # clear_session_state_except_password_correct()
            # st.session_state["last_response"] = "Patient Response: I can't believe I'm in the Emergency Room feeling sick!"
    # Audio response section 
    # Define the URL and headers
    
    if st.session_state.audio_off == False:
        # audio_url = "https://play.ht/api/v2/tts"
        # headers = {
        #     "AUTHORIZATION": f"Bearer {st.secrets['HT_API_KEY']}",
        #     "X-USER-ID": st.secrets["X-USER-ID"],
        #     "accept": "text/event-stream",
        #     "content-type": "application/json",
        # }
        
        # st.write(st.session_state.last_response)
        # st.sidebar.write(response)
        if st.session_state.last_response:
            #     patient_section = extract_patient_response(st.session_state.last_response)
            # st.write(patient_section)
                
                # Define the data
            path_audio = play_audio_eleven_all(st.session_state.last_response, voice=voice)
            
            # data = {
            #     "text": st.session_state.last_response,
            #     "voice": voice,
            # }

            # Send the POST request
            # response_from_audio = requests.post(audio_url, headers=headers, data=json.dumps(data))
            # st.sidebar.write(response_from_audio.text)
            # st.write(f'Audio full: {response_from_audio.text}')
            # st.write(f'Audio url: {response_from_audio.json()}')
            # Print the response
            # link_to_audio = extract_url(response_from_audio.text)
            # st.write(link_to_audio)
            # autoplay_local_audio(path_audio)
    