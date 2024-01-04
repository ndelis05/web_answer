import streamlit as st
import openai
import os  
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain, create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from prompts import *
from openai import OpenAI
import time
from using_docker import using_docker

def process_model_name(model):
    prefix = "openai/"
    if model.startswith(prefix):
        model = model[len(prefix):]
    return model

def update_messages(messages, system_content=None, assistant_content=None, user_content=None):
    """
    Updates a list of message dictionaries with new system, user, and assistant content.

    :param messages: List of message dictionaries with keys 'role' and 'content'.
    :param system_content: Optional new content for the system message.
    :param user_content: Optional new content for the user message.
    :param assistant_content: Optional new content for the assistant message.
    :return: Updated list of message dictionaries.
    """

    # Update system message or add it if it does not exist
    system_message = next((message for message in messages if message['role'] == 'system'), None)
    if system_message is not None:
        if system_content is not None:
            system_message['content'] = system_content
    else:
        if system_content is not None:
            messages.append({"role": "system", "content": system_content})

    # Add assistant message if provided
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    # Add user message if provided
    if user_content is not None:
        messages.append({"role": "user", "content": user_content})

    return messages

def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context, model):
    # st.write('yes the function is being used!')
    messages_blank = []
    messages = update_messages(
        messages = messages_blank, 
        system_content=f'{prefix}; Sample question: {sample_question} Sample response: {sample_answer} Preceding conversation: {history_context}', 
        assistant_content='',
        user_content=my_ask,
        )
    # st.write(messages)
    model2 = process_model_name(model)
    if model2 == model:
        api_key = st.secrets["OPENROUTER_API_KEY"]
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        params = {
            "extra_headers": {
                "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app/",
                "X-Title": 'MediMate GPT and Med Ed',
            }
        }
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
    else:
        api_key = st.secrets["OPENAI_API_KEY"]
        client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
        )
        params = {
            "model": model2,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
    # st.write(f'here are the params: {params}')
    try:    
        completion = client.chat.completions.create(**params)
    except Exception as e:
        st.write(e)
        st.write(f'Here were the params: {params}')
        return None

    placeholder = st.empty()
    full_response = ''
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
            # full_response.append(chunk.choices[0].delta.content)
            placeholder.markdown(full_response)
    placeholder.markdown(full_response)
    return full_response



@st.cache_data
def answer_using_prefix_old(prefix, sample_question, sample_answer, my_ask, temperature, history_context, model, print = True):

    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-1106":
        model = "gpt-3.5-turbo-1106"
    if model == "openai/gpt-4":
        model = "gpt-4-1106-preview"
    if history_context == None:
        history_context = ""
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    if model == "gpt-4-1106-preview" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-1106":
        openai.api_base = "https://api.openai.com/v1/"
        openai.api_key = st.secrets['OPENAI_API_KEY']
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = 500,
        stream = True,   
        )
    else:      
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = st.secrets["OPENROUTER_API_KEY"]
        # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        route = "fallback",
        messages = messages,
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
            "X-Title": "GPT and Med Ed"},
        temperature = temperature,
        max_tokens = 500,
        stream = True,   
        )
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in completion:   
        if print:     
            c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content

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
        if not using_docker:
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.write("*Please contact David Liebovitz, MD if you need an updated password for access.*")
            return False
        else:
            st.session_state["password_correct"] = True
            return True
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




if check_password():
    
    if 'model_bias' not in st.session_state:
        st.session_state.model_bias = "openai/gpt-4"
    
    if 'sample_progress_note' not in st.session_state:
        st.session_state.sample_progress_note = ''
    
    if 'pasted_note' not in st.session_state:
        st.session_state.pasted_note = ''
    
    if 'copied_note' not in st.session_state:
        st.session_state.copied_note = ''

    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'output_history' not in st.session_state:
        st.session_state.output_history = []
                
    if 'mcq_history' not in st.session_state:
        st.session_state.mcq_history = []
        
    st.session_state.temp = 0.5

    # API_O = st.secrets["OPENAI_API_KEY"]
    # Define Streamlit app layout

    st.set_page_config(page_title='Bias Checker', layout = 'centered', page_icon = ':disappointed:', initial_sidebar_state = 'auto')
    st.title("Bias Checker - EARLY DRAFT VERSION")
    st.write("ALPHA version 0.2")
    disclaimer = """**Disclaimer:** This is a early draft tool to identify chart note biases. \n 
2. This tool is not a real doctor. \n    
3. You will not take any medical action based on the output of this tool. \n   
    """
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = openai_api_key
    with st.expander('About Bias Checker - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.write("Last updated 9/26/23")
    with st.expander("Types of Biases (not a complete list)"):
        st.markdown(bias_types)
    
    st.session_state.model_bias = st.selectbox("Model Options", ("openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-1106", "openai/gpt-4", "anthropic/claude-instant-v1", "google/palm-2-chat-bison", "meta-llama/codellama-34b-instruct", "meta-llama/llama-2-70b-chat", "gryphe/mythomax-L2-13b", "nousresearch/nous-hermes-llama2-13b"), index=2)
    if st.session_state.model_bias == "google/palm-2-chat-bison":
        st.warning("The Google model doesn't stream the output, but it's fast. (Will add Med-Palm2 when it's available.)")
        st.markdown("[Information on Google's Palm 2 Model](https://ai.google/discover/palm2/)")
    if st.session_state.model_bias == "openai/gpt-4":
        st.warning("GPT-4 is much better at bias insertion and detection.")
        st.markdown("[Information on OpenAI's GPT-4](https://platform.openai.com/docs/models/gpt-4)")
    if st.session_state.model_bias == "anthropic/claude-instant-v1":
        st.markdown("[Information on Anthropic's Claude-Instant](https://www.anthropic.com/index/releasing-claude-instant-1-2)")
    if st.session_state.model_bias == "meta-llama/llama-2-70b-chat":
        st.markdown("[Information on Meta's Llama2](https://ai.meta.com/llama/)")
    if st.session_state.model_bias == "openai/gpt-3.5-turbo":
        st.markdown("[Information on OpenAI's GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)")
    if st.session_state.model_bias == "openai/gpt-3.5-turbo-1106":
        st.markdown("[Information on OpenAI's GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)")
    if st.session_state.model_bias == "gryphe/mythomax-L2-13b":
        st.markdown("[Information on Gryphe's Mythomax](https://huggingface.co/Gryphe/MythoMax-L2-13b)")
    if st.session_state.model_bias == "meta-llama/codellama-34b-instruct":
        st.markdown("[Information on Meta's CodeLlama](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)")
 
    st.info("Let AI help identify bias 😞 in notes." )
    # st.markdown('[Sample Oncology Notes](https://www.medicaltranscriptionsamplereports.com/hepatocellular-carcinoma-discharge-summary-sample/)')
    
    col1, col2 = st.columns(2)
    with col1:
        task = st.radio("What would you like to do?", ("Generate a sample note and check for bias", "Paste a sample note to check for bias", "Upload a batch of notes to check for bias",))

    if task == "Generate a sample note and check for bias":
        st.sidebar.warning("This is an EARLY PHASE TOOL undergoing significant updates soon. Eventually, it will generate biased yet realistic note examples for us all to learn from.")    
        st.warning("Enter details into the sidebar on the left and use the buttons to generate response")
        desired_note_content = st.sidebar.text_input("Please enter a specialty and diagnoses for your generated progress note:")
        patient_attributes = st.sidebar.text_input("Please enter one or more patient attributes you would like to use for your note (e.g., 35 yo white male with obesity): ")
        desired_note_bias = st.sidebar.text_input("Please enter one or more biases you would like to insert within your sample note: ")
        desired_note_prompt = desired_note_prompt.format(desired_note_content=desired_note_content, patient_attributes = patient_attributes, desired_note_bias=desired_note_bias)
        
        if st.sidebar.button("Step 1: Generate a Sample Progress Note"):

            st.session_state.sample_progress_note = answer_using_prefix(
                    biased_note_generator_context, 
                    "", 
                    '', 
                    desired_note_prompt, 
                    st.session_state.temp, 
                    history_context="",
                    model = st.session_state.model_bias,
                    )
            
        with st.expander("Your Sample Progress Note"):
            st.write(st.session_state.sample_progress_note)

        if st.session_state.sample_progress_note != '':
            if st.sidebar.button("Step 2: Assess for Bias"):
            
                bias_assessment = answer_using_prefix(
                bias_detection_prompt, 
                biased_note_example, 
                bias_report_example,
                st.session_state.sample_progress_note, 
                st.session_state.temp, 
                history_context="",
                model = st.session_state.model_bias,
                )   
    if task == "Paste a sample note to check for bias":
        st.warning("No PHI!!!: Enter details into the text box")
        st.session_state.pasted_note = st.text_area("Please paste a sample note to check for bias: ")
        if st.button("Assess for Bias"):
            bias_assessment = answer_using_prefix(
                bias_detection_prompt, 
                biased_note_example, 
                bias_report_example,
                st.session_state.pasted_note, 
                st.session_state.temp, 
                history_context="",
                model = st.session_state.model_bias,
                )
            
    if task == "Upload a batch of notes to check for bias":
        st.warning("No PHI!!! Enter details into the sidebar on the left and use the buttons to generate response")
        st.session_state.uploaded_file = st.sidebar.file_uploader("Please upload a batch of notes to check for bias: ")
        if st.button("Assess for Bias"):
            if st.session_state.uploaded_file is not None:
                st.session_state.copied_note = st.session_state.uploaded_file.getvalue().decode("utf-8")
                bias_assessment = answer_using_prefix(
                    bias_detection_prompt, 
                    biased_note_example, 
                    bias_report_example,
                    st.session_state.copied_note, 
                    st.session_state.temp, 
                    history_context="",
                    model = st.session_state.model_bias,
                    )
            else:
                st.warning("Please upload a file to check for bias")
        with st.expander("Your Uploaded Notes"):
            st.write(st.session_state.copied_note)