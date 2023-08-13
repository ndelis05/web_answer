import streamlit as st
import openai
import requests
import time
import json
import os
from prompts import *
from functions import *





def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except KeyError:
        
        try:
            api_key = os.environ["OPENAI_API_KEY"]
            # If the API key is already set, don't prompt for it again
            return api_key
        except KeyError:        
            # If the secret is not found, prompt the user for their API key
            st.warning("Oh, dear friend of mine! It seems your API key has gone astray, hiding in the shadows. Pray, reveal it to me!")
            api_key = getpass.getpass("Please, whisper your API key into my ears: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Save the API key as a secret
            # st.secrets["my_api_key"] = api_key
            return api_key
    
    return api_key


def check_password():
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

@st.cache_resource
def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_key = os.environ['OPENAI_API_KEY']
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
    # model = 'gpt-3.5-turbo',
    model = st.session_state.model,
    messages = messages,
    temperature = temperature,
    stream = True,   
    )
    
    
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:        
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content


if 'dc_history' not in st.session_state:
    st.session_state.dc_history = []

if 'annotate_history' not in st.session_state:
    st.session_state.annotate_history = []

if 'history' not in st.session_state:
    st.session_state.history = []

if 'output_history' not in st.session_state:
    st.session_state.output_history = []
            

            
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    
if 'model' not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"
    
if 'temp' not in st.session_state:
    st.session_state.temp = 0.3

if check_password():

    st.set_page_config(page_title='Basic Answers', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
    st.title("GPT and Medical Education")
    st.write("ALPHA version 0.3")
    os.environ['OPENAI_API_KEY'] = fetch_api_key()

    disclaimer = """**Disclaimer:** This is a tool to assist education regarding artificial intelligence. Your use of this tool accepts the following:   
    1. This tool does not generate validated medical content. \n 
    2. This tool is not a real doctor. \n    
    3. You will not take any medical action based on the output of this tool. \n   
    """
    with st.expander('About GPT and Med Ed - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.session_state.model = st.radio("Select model - leave default for now", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"), index=0)
        st.session_state.temp = st.slider("Select temperature", 0.0, 1.0, 0.3, 0.01)
        st.write("Last updated 8/12/23")



    tab1, tab2 = st.tabs(["Learn", "Patient Communication", ])

        
    with tab1:
    
        persona = st.radio("Select teaching persona", ("Teacher 1 (academic)", "Teacher 2 (analogies)", "Minimal prompt"), index=0)
        my_ask = st.text_area('Enter a topic: (e.g., RAAS, Frank-Starling, sarcoidosis, etc.)', height=100, key="my_ask")
        my_ask = my_ask.replace("\n", " ")
        my_ask = "Teach me about: " + my_ask

        if persona == "Minimal prompt":
            system_context = base_teacher
        elif persona == "Teacher 1 (academic)":
            system_context = teacher1
        elif persona == "Teacher 2 (analogies)":
            system_context = teacher2
        
        if st.button("Enter"):
            openai.api_key = os.environ['OPENAI_API_KEY']
            st.session_state.history.append(my_ask)
            history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
            output_text = answer_using_prefix(system_context, sample_question, sample_response, my_ask, st.session_state.temp, history_context=history_context)
            # st.session_state.my_ask = ''
            # st.write("Answer", output_text)
            
            # st.write(st.session_state.history)
            # st.write(f'Me: {my_ask}')
            # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
            # st.write(list(output_text))
            # st.session_state.output_history.append((output_text['choices'][0]['message']['content']))
            st.session_state.output_history.append((output_text))
            
        if st.button("Clear Memory (when you don't want to send prior context)"):
            st.session_state.history = []
            st.session_state.output_history = []
            st.write("Memory cleared")
        
        tab1_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
        with st.expander("View or Download Thread", expanded=False):
            for i in range(len(st.session_state['output_history'])-1, -1, -1):
                st.info(st.session_state["history"][i],icon="🧐")
                st.success(st.session_state["output_history"][i], icon="🤖")
                tab1_download_str.append(st.session_state["history"][i])
                tab1_download_str.append(st.session_state["output_history"][i])
            tab1_download_str = [disclaimer] + tab1_download_str 
            
            # Can throw error - requires fix
            tab1_download_str = '\n'.join(tab1_download_str)
            if tab1_download_str:
                st.download_button('Download',tab1_download_str, key = "Conversation_Thread")
                
    with tab2:
        st.subheader("Patient Communication")
        col1, col2 = st.columns(2)
        with col2:
            health_literacy_level = st.radio("Output optimized for:", ("Low Health Literacy", "High Health Literacy"))
   

        with col1:
            task = st.radio("What do you want to do?", ("Generate discharge instructions", "Annotate a patient result"))

        if task == "Generate discharge instructions":
            answer = ''
            start_time = time.time()
            surg_procedure = st.text_area("Please enter the procedure performed and any special concerns.")
            dc_meds = st.text_area("Please enter the discharge medications.")
            dc_instructions_context = f'Generate discharge instructions for a patient as if it is authored by a physician for her patient with {health_literacy_level} with this {surg_procedure} on {dc_meds}'
            if st.button("Generate Discharge Instructions"):
                try:
                    dc_text = answer_using_prefix(
                        dc_instructions_prompt, 
                        procedure_example, 
                        dc_instructions_example, 
                        surg_procedure, 
                        st.session_state.temp, 
                        history_context="",
                        )
                    st.session_state.dc_history.append((dc_text))  
                except:
                    st.write("Error - please try again")
                      

        
            dc_download_str = []
                
                # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                # Display the conversation history using an expander, and allow the user to download it
            with st.expander("View or Download Instructions", expanded=False):
                for i in range(len(st.session_state['dc_history'])-1, -1, -1):
                    st.info(st.session_state["dc_history"][i],icon="🧐")
                    st.success(st.session_state["dc_history"][i], icon="🤖")
                    dc_download_str.append(st.session_state["dc_history"][i])
                    
                dc_download_str = [disclaimer] + dc_download_str 
                
                
                dc_download_str = '\n'.join(dc_download_str)
                if dc_download_str:
                    st.download_button('Download',dc_download_str, key = "DC_Thread")        
                    

        if task == "Annotate a patient result":
            sample_report1 = st.sidebar.radio("Try a sample report:", ("Text box for your own content", "Sample 1 (lung CT)", "Sample 2 (ECG)", ))
            if sample_report1 == "Sample 1 (lung CT)":
                submitted_result = report1
                with col1:
                    st.write(report1)
            elif sample_report1 == "Sample 2 (ECG)":
                submitted_result = report2
                with col1:
                    st.write(report2)
            elif sample_report1 == "Text box for your own content":           
                with col1:                
                    submitted_result = st.text_area("Paste your result content here without PHI.", height=600)
            
            
            report_prompt = f'Generate a brief reassuring summary as if it is authored by a physician for her patient with {health_literacy_level} with this {submitted_result}. When appropriate emphasize that the findings are not urgent and you are happy to answer any questions at the next visit. '

            
            if st.button("Generate Annotation"):
                try:
                    with col2:
                        annotate_text = answer_using_prefix(
                            annotate_prompt, 
                            report1, 
                            annotation_example,
                            report_prompt, 
                            st.session_state.temp, 
                            history_context="",
                            )                    

                        st.session_state.annotate_history.append((annotate_text))
                except:
                    with col2:
                        st.write("API busy. Try again - better error handling coming. :) ")
                        st.stop()
            
                    
        
            annotate_download_str = []
                
                # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                # Display the conversation history using an expander, and allow the user to download it
            with st.expander("View or Download Annotations", expanded=False):
                for i in range(len(st.session_state['annotate_history'])-1, -1, -1):
                    st.info(st.session_state["annotate_history"][i],icon="🧐")
                    st.success(st.session_state["annotate_history"][i], icon="🤖")
                    annotate_download_str.append(st.session_state["annotate_history"][i])
                    
                annotate_download_str = [disclaimer] + annotate_download_str 
                
                
                annotate_download_str = '\n'.join(annotate_download_str)
                if annotate_download_str:
                    st.download_button('Download',annotate_download_str, key = "Annotate_Thread")        
            
            
