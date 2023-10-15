import streamlit as st
import openai
from prompts import *
import time

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
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

def compress_summary(summary):
    openai.api_base = "https://api.openai.com/v1/"
    openai.api_key = st.secrets['OPENAI_API_KEY']
    with st.spinner("Compressing messsages for summary..."):
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-16k",
            temperature = 0.3,
            messages = [
                {
                    "role": "system",
                    "content": "Generate a brief summary of this message thread and where it left off at the end. Do not repeat details; just summarize the main points."
                },
                {
                    "role": "user",
                    "content": summary
                }
            ],
            max_tokens = 300, 
        )
    return completion['choices'][0]['message']['content']
    
    
@st.cache_data
def interactive_chat(messages, temperature, model, print = True):

    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    # st.write(f'question: {history_context + my_ask}')
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        openai.api_base = "https://api.openai.com/v1/"
        openai.api_key = st.secrets['OPENAI_API_KEY']
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = 1000,
        stream = True,   
        )
    else:      
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = st.secrets["OPENROUTER_API_KEY"]
        st.write(st.session_state.messages)
        # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
        completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
        # model = 'gpt-3.5-turbo',
        model = model,
        route = "fallback",
        messages = messages,
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
            "X-Title": "GPT and Med Ed"},
        temperature = temperature,
        max_tokens = 1000,
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
    if model == "google/palm-2-chat-bison":
        
        st.markdown(full_answer)
    return full_answer # Change how you access the message content

def summarize_messages(messages):
    total_chars = sum(len(message['content']) for message in messages)
    # st.write(f'total characters are: {total_chars}')
    if total_chars > 7000:
        if len(messages) >= 3:
            # Keep the first message and the most recent two messages intact
            first_message = messages[0]
            last_two_messages = messages[-2:]
            
            # Summarize all other messages
            other_messages = messages[1:-2]
            summary_content = 'Prior messages: ' + ' '.join(message['content'] for message in other_messages)
            summary_content = compress_summary(summary_content)
            summary_message = {'role': 'assistant', 'content': summary_content}
            
            # Combine the messages
            messages = [first_message, summary_message] + last_two_messages
        elif len(messages) == 2:
                # If there are only two messages, summarize them into one
                summary_content = 'Prior messages: ' + ' '.join(message['content'] for message in messages)
                summary_content = compress_summary(summary_content)
                messages = [{'role': 'assistant', 'content': summary_content}]
        else:
            # If there's only one message, no need to summarize
            pass
    
    return messages



st.set_page_config(page_title='Interactive Teacher for Medical Topics!', layout = 'centered', page_icon = '🧑🏾‍🏫', initial_sidebar_state = 'auto')
st.title("Interactive Teacher for Foundational Medical Topics!")

if check_password():
    
    proceed = st.checkbox("Acknowledge - this tool should be used for foundational knowledge only. Do not use to learn about the latest treatments.")
    
    if proceed:
        st.write("👇🏾Enter your question or topic at the bottom.👇🏾 Thank you for using this tool responsibly.")

        with st.expander("Settings and ℹ️ About this app"):
            st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.3, 0.01)
            st.session_state.model = st.selectbox("Model Options", ("openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-16k",  "openai/gpt-4", "anthropic/claude-instant-v1", "google/palm-2-chat-bison",), index=2)
            st.write("ℹ️ Do not use to learn about the latest treatments. Use to bulk up your foundational knowledge where GPT is most reliable.")
        
            # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{'role': 'system', 'content': interactive_teacher},]
            
        for message in st.session_state.messages:
            if message['role'] != 'system':
                with st.chat_message(message["role"]):
                    st.markdown(message["content"], unsafe_allow_html=True)
                    

                
        if prompt := st.chat_input("What topic do you want to learn about?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
            
            with st.spinner("Thinking..."):   
                # for response in openai.ChatCompletion.create(
                #         model=st.session_state["openai_model"],
                #         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                #         headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
                #             "X-Title": "GPT and Med Ed"},
                #         stream=True,
                #     ):
                #     full_response += response.choices[0].delta.get("content", "")
                st.session_state.messages = summarize_messages(st.session_state.messages)
                full_response = interactive_chat(st.session_state.messages, st.session_state.temp, st.session_state.model)
                
            if st.button("Clear Memory (when you want to switch topics)"):
                st.session_state.messages = []
                st.write("Memory cleared")
                full_response = ""
                st.session_state.messages = [{'role': 'system', 'content': interactive_teacher},]
            
            # message_placeholder.markdown(full_response + "▌")
            # message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
