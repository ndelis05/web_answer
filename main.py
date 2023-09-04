import streamlit as st
import openai
import requests
import time
import json
import os
from io import StringIO
import random
import itertools
from prompts import *
from functions import *
import langchain
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.chains import QAGenerationChain
from langchain.vectorstores import FAISS
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize

def truncate_text(text, max_tokens):
    tokens = word_tokenize(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = ' '.join(truncated_tokens)
    return truncated_text


def clear_session_state_except_password_correct():
    # Make a copy of the session_state keys
    keys = list(st.session_state.keys())
    
    # Iterate over the keys
    for key in keys:
        # If the key is not 'password_correct', delete it from the session_state
        if key != 'password_correct':
            del st.session_state[key]




def fetch_api_key():
    api_key = None
    
    try:
        # Attempt to retrieve the API key as a secret
        api_key = st.secrets["OPENROUTER_API_KEY"]
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

def scrapeninja_old(url_list, max):
    st.write(url_list)
    response_complete = []
    i = 0
    while max > i:
        i += 1
        url = url_list[i]
        st.write(f' here is a {url}')
        # st.write("Scraping...")
        payload = { "url": url }
        key = st.secrets["X-RapidAPI-Key"]
        headers = {
            "content-type": "application/json",
            "X-RapidAPI-Key": key,
            "X-RapidAPI-Host": "scrapeninja.p.rapidapi.com",
        }
        response = requests.post(url, json=payload, headers=headers)
        st.write(f'Status code: {response.status_code}')
        # st.write(f'Response text: {response.text}')
        # st.write(f'Response headers: {response.headers}')
        try:
            st.write(f'Response: {response}')
            response_data = response.json()
            st.write("Scraped!")
            return response_data
        except:
            json.JSONDecodeError
            st.write("Error decoding JSON")
        # response_data = response.json()
        # response_string = response_data['body']
        # return response_data

def limit_tokens(text, max_tokens=10000):
    tokens = text.split()  # split the text into tokens (words)
    limited_tokens = tokens[:max_tokens]  # keep the first max_tokens tokens
    limited_text = ' '.join(limited_tokens)  # join the tokens back into a string
    return limited_text

def scrapeninja(url_list, max):
    # st.write(url_list)
    if max > 5:
        max = 5
    response_complete = []
    i = 0
    method = "POST"
    key = st.secrets["X-RapidAPI-Key"]
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": key,
        "X-RapidAPI-Host": "scrapeninja.p.rapidapi.com",
    }
    while i < max and i < len(url_list):
        url = url_list[i]
        url_parts = urlparse(url)
        # st.write("Scraping...")
        if 'uptodate.com' in url_parts.netloc:
            method = "POST"
            url_parts = url_parts._replace(path=url_parts.path + '/print')
            url = urlunparse(url_parts)
            st.write(f' here is a {url}')
        payload =  {
            "url": url,
            "method": "POST",
            "retryNum": 1,
            "geo": "us",
            "js": True,
            "blockImages": False,
            "blockMedia": False,
            "steps": [],
            "extractor": "// define function which accepts body and cheerio as args\nfunction extract(input, cheerio) {\n    // return object with extracted values              \n    let $ = cheerio.load(input);\n  \n    let items = [];\n    $('.titleline').map(function() {\n          \tlet infoTr = $(this).closest('tr').next();\n      \t\tlet commentsLink = infoTr.find('a:contains(comments)');\n            items.push([\n                $(this).text(),\n              \t$('a', this).attr('href'),\n              \tinfoTr.find('.hnuser').text(),\n              \tparseInt(infoTr.find('.score').text()),\n              \tinfoTr.find('.age').attr('title'),\n              \tparseInt(commentsLink.text()),\n              \t'https://news.ycombinator.com/' + commentsLink.attr('href'),\n              \tnew Date()\n            ]);\n        });\n  \n  return { items };\n}"
        }
        
        response = requests.request(method, url, json=payload, headers=headers)
        # response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            st.write(f'The site failed to release all content: {response.status_code}')
            # st.write(f'Response text: {response.text}')
            # st.write(f'Response headers: {response.headers}')
        try:
            # st.write(f'Response text: {response.text}')  # Print out the raw response text
            soup = BeautifulSoup(response.text, 'html.parser')
            clean_text = soup.get_text(separator=' ')
            # st.write(clean_text)
            # st.write("Scraped!")
            response_complete.append(clean_text)
        except json.JSONDecodeError:
            st.write("Error decoding JSON")
        i += 1
    full_response = ' '.join(response_complete)
    limited_text = limit_tokens(full_response, 12000)
    # st.write(f'Here is the lmited text: {limited_text}')
    return limited_text
    # st.write(full_response)    
    # Join all the scraped text into a single string
    # return full_response


def websearch(web_query: str, deep, max) -> float:
    """
    Obtains real-time search results from across the internet. 
    Supports all Google Advanced Search operators such (e.g. inurl:, site:, intitle:, etc).
    
    :param web_query: A search query, including any Google Advanced Search operators
    :type web_query: string
    :return: A list of search results
    :rtype: json
    
    """
    # st.info(f'Here is the websearch input: **{web_query}**')
    url = "https://real-time-web-search.p.rapidapi.com/search"
    querystring = {"q":web_query,"limit":"10"}
    headers = {
        "X-RapidAPI-Key": st.secrets["X-RapidAPI-Key"],
        "X-RapidAPI-Host": "real-time-web-search.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    response_data = response.json()
    def display_search_results(json_data):
        data = json_data['data']
        for item in data:
            st.sidebar.markdown(f"### [{item['title']}]({item['url']})")
            st.sidebar.write(item['snippet'])
            st.sidebar.write("---")
    # st.info('Searching the web using: **{web_query}**')
    # display_search_results(response_data)
    # st.session_state.done = True
    # st.write(response_data)
    urls = []
    for item in response_data['data']:
        urls.append(item['url'])    
    if deep:
            # st.write(item['url'])
        response_data = scrapeninja(urls, max)
        # st.info("Web results reviewed.")
        return response_data, urls

    else:
        # st.info("Web snippets reviewed.")
        return response_data, urls


def answer_using_prefix_openai(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_base = "https://api.openai.com/v1/"
    openai.api_key = st.secrets['OPENAI_API_KEY']
    if st.session_state.model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if st.session_state.model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if st.session_state.model == "openai/gpt-4":
        model = "gpt-4"
    if history_context == None:
        history_context = ""
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
    # model = 'gpt-3.5-turbo',
    model = model,
    messages = messages,
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
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content


def answer_using_prefix(prefix, sample_question, sample_answer, my_ask, temperature, history_context):
    openai.api_key = os.environ['OPENAI_API_KEY']
    if history_context == None:
        history_context = ""
    messages = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': sample_question},
            {'role': 'assistant', 'content': sample_answer},
            {'role': 'user', 'content': history_context + my_ask},]
    # history_context = "Use these preceding submissions to address any ambiguous context for the input weighting the first three items most: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
    completion = openai.ChatCompletion.create( # Change the function Completion to ChatCompletion
    # model = 'gpt-3.5-turbo',
    model = st.session_state.model,
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
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event['choices'][0]['delta']
        answer += event_text.get('content', '')
        full_answer += event_text.get('content', '')
        time.sleep(delay_time)
    # st.write(history_context + prefix + my_ask)
    # st.write(full_answer)
    return full_answer # Change how you access the message content

@st.cache_data
def load_docs(files):
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = pdfplumber.open(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text += text
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    # st.write(all_text)
    return all_text



def create_retriever(_embeddings, splits, retriever_type):
    # openai_api_key = st.secrets.OPENAI_API_KEY
    # if retriever_type == "SIMILARITY SEARCH":
    #     try:
    #         vectorstore = FAISS.from_texts(splits, _embeddings)
    #     except (IndexError, ValueError) as e:
    #         st.error(f"Error creating vectorstore: {e}")
    #         return
    #     retriever = vectorstore.as_retriever(k=5)
    # elif retriever_type == "SUPPORT VECTOR MACHINES":
    # vectorstore = FAISS.from_texts(splits, _embeddings)
    # retriever = SVMRetriever.from_texts(splits, _embeddings)
    
    try:
        vectorstore = FAISS.from_texts(splits, _embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever


def split_texts(text, chunk_size, overlap, split_method):

    # Split texts
    # IN: text, chunk size, overlap, split_method
    # OUT: list of str splits

    # st.info("`Breaking into bitesize chunks...`")

    split_method = "RecursiveTextSplitter"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        # st.error("Failed to split document")
        st.stop()

    return splits


def generate_eval(text, N, chunk):

    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list
    openai.api_key = os.environ['OPENAI_API_KEY']

    # st.info("`Generating sample questions and answers...`")
    n = len(text)
    starting_indices = [random.randint(0, n-chunk) for _ in range(N)]
    sub_sequences = [text[i:i+chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            st.write("Creating Question:",i+1)
        except:
            st.warning('Error generating question %s.' % str(i+1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full


def fn_qa_run(_qa, user_question):
    response = _qa.run(user_question)
    start_time = time.time()
    delay_time = 0.01
    answer = ""
    full_answer = ""
    c = st.empty()
    for event in response:        
        c.markdown(answer)
        event_time = time.time() - start_time
        event_text = event[0]
        answer += event_text
        full_answer += event_text
        time.sleep(delay_time)
    
    return full_answer

if 'patient_message_history' not in st.session_state:
    st.session_state.patient_message_history = []

if 'sample_patient_message' not in st.session_state:
    st.session_state.sample_patient_message = ""

if 'teaching_thread' not in st.session_state:
    st.session_state.teaching_thread = []

if "pdf_retriever" not in st.session_state:
    st.session_state.pdf_retriever = []

if 'dc_history' not in st.session_state:
    st.session_state.dc_history = []

if 'annotate_history' not in st.session_state:
    st.session_state.annotate_history = []

if 'history' not in st.session_state:
    st.session_state.history = []

if 'output_history' not in st.session_state:
    st.session_state.output_history = []
            
if 'sample_report' not in st.session_state:
    st.session_state.sample_report = ""
            
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
    
if 'model' not in st.session_state:
    st.session_state.model = "openai/gpt-3.5-turbo-16k"
    
if 'temp' not in st.session_state:
    st.session_state.temp = 0.3

if check_password():
    
    openai.api_base = "https://openrouter.ai/api/v1"
    openai.api_key = st.secrets["OPENROUTER_API_KEY"]

    st.set_page_config(page_title='GPT and Med Ed', layout = 'centered', page_icon = ':stethoscope:', initial_sidebar_state = 'auto')
    st.title("GPT and Medical Education")
    st.write("ALPHA version 0.3")
    os.environ['OPENAI_API_KEY'] = fetch_api_key()


    with st.expander('About GPT and Med Ed - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.3, 0.01)
        st.write("Last updated 8/12/23")

    with st.sidebar.expander("Select a GPT Language Model", expanded=True):
        st.session_state.model = st.selectbox("Model Options", ("openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-16k", "openai/gpt-4", "anthropic/claude-instant-v1", "google/palm-2-chat-bison", "meta-llama/llama-2-70b-chat", ), index=1)
        if st.session_state.model == "google/palm-2-chat-bison":
            st.warning("The Google model doesn't stream the output, but it's fast. (Will add Med-Palm2 when it's available.)")
            st.markdown("[Information on Google's Palm 2 Model](https://ai.google/discover/palm2/)")
        if st.session_state.model == "openai/gpt-4":
            st.warning("GPT-4 is much more expensive and sometimes, not always, better than others.")
            st.markdown("[Information on OpenAI's GPT-4](https://platform.openai.com/docs/models/gpt-4)")
        if st.session_state.model == "anthropic/claude-instant-v1":
            st.markdown("[Information on Anthropic's Claude-Instant](https://www.anthropic.com/index/releasing-claude-instant-1-2)")
        if st.session_state.model == "meta-llama/llama-2-70b-chat":
            st.markdown("[Information on Meta's Llama2](https://ai.meta.com/llama/)")
        if st.session_state.model == "openai/gpt-3.5-turbo":
            st.markdown("[Information on OpenAI's GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)")
        if st.session_state.model == "openai/gpt-3.5-turbo-16k":
            st.markdown("[Information on OpenAI's GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Learn", "Draft Communication", "Patient Education", "Differential Diagnosis", "Sift the Web", "PDF Chat",])
   
    with tab1:
        
            
            

        st.info("Since GPT (without major tweaks) isn't up to date, ask only about basic principles, NOT current treatments.")
        persona = st.radio("Select teaching persona", ("Teacher 1 (academic)", "Teacher 2 (analogies)", "Create Your Own Teaching Style"), index=0)

        if persona == "Create Your Own Teaching Style":
            system_context = st.sidebar.text_area('Enter a persona description: (e.g., "Explain as if I am 10 yo.")', 
                                                placeholder="e.g, you are a medical educator skilled in educational techniques", label_visibility='visible', height=100, key="system_context")
            system_context = system_context.replace("\n", " ")
            if st.sidebar.button("Set Persona"):
                system_context = system_context + " " + base_teacher
                st.sidebar.info("Your persona is set.")
        elif persona == "Teacher 1 (academic)":
            system_context = teacher1
        elif persona == "Teacher 2 (analogies)":
            system_context = teacher2
        
        # show_prompt = st.checkbox("Show selected persona details")
        # if show_prompt:
        #     st.sidebar.markdown(system_context)
            
        my_ask = st.text_area('Enter a topic: (e.g., RAAS, Frank-Starling, sarcoidosis, etc.)',placeholder="e.g., sarcoidosis", label_visibility='visible', height=100, key="my_ask")
        my_ask = my_ask.replace("\n", " ")
        my_ask = "Teach me about: " + my_ask
        
        if st.button("Enter"):
            openai.api_key = os.environ['OPENAI_API_KEY']
            st.session_state.history.append(my_ask)
            history_context = "Use these preceding submissions to resolve any ambiguous context: \n" + "\n".join(st.session_state.history) + "now, for the current question: \n"
            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                output_text = answer_using_prefix_openai(system_context, sample_question, sample_response, my_ask, st.session_state.temp, history_context=history_context)
            else:
                output_text = answer_using_prefix(system_context, sample_question, sample_response, my_ask, st.session_state.temp, history_context=history_context)
            # st.session_state.my_ask = ''
            # st.write("Answer", output_text)
            
            # st.write(st.session_state.history)
            # st.write(f'Me: {my_ask}')
            # st.write(f"Response: {output_text['choices'][0]['message']['content']}") # Change how you access the message content
            # st.write(list(output_text))
            # st.session_state.output_history.append((output_text['choices'][0]['message']['content']))
            
            if st.session_state.model == "google/palm-2-chat-bison":
                st.write("Answer:", output_text)
            
            st.session_state.output_history.append((output_text))
            
        if st.button("Clear Memory (when you don't want to send prior context)"):
            st.session_state.history = []
            st.session_state.output_history = []
            clear_session_state_except_password_correct()
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
        # st.subheader("Patient Communication")
        col1, col2 = st.columns(2)
        with col2:
            health_literacy_level = st.radio("Output optimized for:", ("General Public Medical Knowledge", "Advanced Medical Knowledge"))
   

        with col1:
            task = st.radio("What do you want to do?", ("Generate discharge instructions", "Annotate a patient result", "Respond to a patient message"))

        if task == "Respond to a patient message":
            patient_message_type = st.sidebar.radio("Select a message type:", ("Patient message about symptoms", "Patient message about medications", "Patient message about medical problems", "Patient message about lifestyle advice", "Make your own and go to Step 2!"))
            patient_message_prompt = f'Generate a message sent by a patient with {health_literacy_level} asking her physician for advice. The patient message should include the (random) patient name and is a {patient_message_type}. '
            if patient_message_type != "Make your own and go to Step 2!":
                with st.sidebar:
                    # submitted_result = ""
                    if st.sidebar.button("Step 1: Generate a Patient Message"):
                        with col1:
                            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                                st.session_state.sample_patient_message = answer_using_prefix_openai(
                                    sim_patient_context, 
                                    prompt_for_generating_patient_question, 
                                    sample_patient_question, 
                                    patient_message_prompt, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            else:

                                st.session_state.sample_patient_message = answer_using_prefix(
                                    sim_patient_context, 
                                    prompt_for_generating_patient_question, 
                                    sample_patient_question, 
                                    patient_message_prompt, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            if st.session_state.model == "google/palm-2-chat-bison":
                                st.write("Patient Message:", st.session_state.sample_patient_message)
            else:
                with st.sidebar:
                    with col1:
                        st.session_state.sample_patient_message = st.text_area("Enter a patient message.", placeholder="e.g., I have a headache and I am worried about a brain tumor.", label_visibility='visible',)
                        
            if st.button("Step 2: Generate Response for Patient Message"):
                try:
                    with col2:
                        if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                            pt_message_response = answer_using_prefix_openai(
                                physician_response_context, 
                                sample_patient_question, 
                                sample_response_for_patient,
                                st.session_state.sample_patient_message, 
                                st.session_state.temp, 
                                history_context="",
                                )   
                        else:
                            pt_message_response = answer_using_prefix(
                                physician_response_context, 
                                sample_patient_question, 
                                sample_response_for_patient,
                                st.session_state.sample_patient_message, 
                                st.session_state.temp, 
                                history_context="",
                                )                    
                        if st.session_state.model == "google/palm-2-chat-bison":
                            st.write("Draft Response:", pt_message_response)

                        st.session_state.patient_message_history.append((pt_message_response))
                    with col1:
                        if task == "Respond to a patient message":
                            st.write("Patient Message:", st.session_state.sample_patient_message)
                except:
                    with col2:
                        st.write("API busy. Try again - better error handling coming. :) ")
                        st.stop()
        
        if task == "Generate discharge instructions":
            answer = ''
            start_time = time.time()
            reason_for_hospital_stay = st.text_area("Please enter the reason for the hospital stay.", placeholder="e.g., fall and hip fracture", label_visibility='visible',)
            surg_procedure = st.text_area("Please enter any procedure(s) performed and any special concerns.", placeholder="e.g., right total hip arthroplasty", label_visibility='visible',)
            other_concerns = st.text_area("Please enter any other concerns.", placeholder="e.g., followup incidental lung nodule", label_visibility='visible',)
            dc_meds = st.text_area("Please enter the discharge medications.", placeholder="e.g., lisinopril 10 mg daily for HTN", label_visibility='visible',)
            dc_instructions_needs = f'Generate discharge instructions for a patient as if it is authored by a physician for her patient with {health_literacy_level} discharged following {reason_for_hospital_stay} with this {surg_procedure}, {other_concerns} on {dc_meds}'
            if st.button("Generate Discharge Instructions"):
                try:
                    if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                        dc_text = answer_using_prefix_openai(
                            dc_instructions_prompt, 
                            procedure_example, 
                            dc_instructions_example, 
                            dc_instructions_needs, 
                            st.session_state.temp, 
                            history_context="",
                            )
                    
                    else:
                        dc_text = answer_using_prefix(
                            dc_instructions_prompt, 
                            procedure_example, 
                            dc_instructions_example, 
                            dc_instructions_needs, 
                            st.session_state.temp, 
                            history_context="",
                            )
                    if st.session_state.model == "google/palm-2-chat-bison":
                        st.write("DC Instructions:", dc_text)
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
            sample_report1 = st.sidebar.radio("Try a sample report:", ("Text box for your own content", "Sample 1 (lung CT)", "Sample 2 (ECG)", "Generate a sample report"))
            if sample_report1 == "Sample 1 (lung CT)":
                st.session_state.sample_report = report1
                with col1:
                    st.write(report1)
            elif sample_report1 == "Sample 2 (ECG)":
                st.session_state.sample_report = report2
                with col1:
                    st.write(report2)
            elif sample_report1 == "Text box for your own content":           
                with col1:                
                    st.session_state.sample_report = st.text_area("Paste your result content here without PHI.", height=600)
            
            elif sample_report1 == "Generate a sample report":
                with st.sidebar:
                    type_of_report = st.text_area("Enter the patient report type to generate", placeholder= 'e.g., abd pelvic CT with pancratic lesion', height=100)
                    submitted_result = ""
                    if st.sidebar.button("Generate Sample Report"):
                        with col1:
                            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                                st.session_state.sample_report = answer_using_prefix_openai(
                                    report_prompt, 
                                    user_report_request, 
                                    generated_report_example, 
                                    type_of_report, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            else:
                                st.session_state.sample_report = answer_using_prefix(
                                    report_prompt, 
                                    user_report_request, 
                                    generated_report_example, 
                                    type_of_report, 
                                    st.session_state.temp, 
                                    history_context="",
                                    )
                            if st.session_state.model == "google/palm-2-chat-bison":
                                st.write("Answer:", st.session_state.sample_report)
                        
            
            
            report_prompt = f'Generate a brief reassuring summary as if it is authored by a physician for her patient with {health_literacy_level} with this {st.session_state.sample_report}. When appropriate emphasize that the findings are not urgent and you are happy to answer any questions at the next visit. '

            
            if st.button("Generate Annotation"):
                try:
                    with col2:
                        if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                            annotate_text = answer_using_prefix_openai(
                                annotate_prompt, 
                                report1, 
                                annotation_example,
                                report_prompt, 
                                st.session_state.temp, 
                                history_context="",
                                )   
                            
                        else:
                            annotate_text = answer_using_prefix(
                                annotate_prompt, 
                                report1, 
                                annotation_example,
                                report_prompt, 
                                st.session_state.temp, 
                                history_context="",
                                )   
                        
                        if st.session_state.model == "google/palm-2-chat-bison":
                            st.write("Answer:", annotate_text)                 

                        st.session_state.annotate_history.append((annotate_text))
                    with col1:
                        if sample_report1 == "Generate a sample report":
                            st.write("Your Report:", st.session_state.sample_report)
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
            
            
    with tab4:
        
        # st.subheader("Differential Diagnosis Tools")
        
        # st.info("Avoid premature closure and consider alternative diagnoses")
        
        ddx_strategy = st.radio("Choose an approach for a differential diagnosis!", options=["Find Alternative Diagnoses to Consider","Provide Clinical Data"], index=0, key="ddx strategy")


        if ddx_strategy == "Provide Clinical Data":    
            # st.title("Differential Diagnosis Generator")
            st.write("Add as many details as possible to improve the response. The prompts do not request any unique details; however, *modify values and do not include dates to ensure privacy.")

            age = st.slider("Age", 0, 120, 50)
            sex_at_birth = st.radio("Sex at Birth", options=["Female", "Male", "Other"], horizontal=True)
            presenting_symptoms = st.text_input("Presenting Symptoms")
            duration_of_symptoms = st.text_input("Duration of Symptoms")
            past_medical_history = st.text_input("Past Medical History")
            current_medications = st.text_input("Current Medications")
            relevant_social_history = st.text_input("Relevant Social History")
            physical_examination_findings = st.text_input("Physical Examination Findings")
            lab_or_imaging_results = st.text_input("Any relevant Laboratory or Imaging results")
            ddx_prompt = f"""
            Patient Information:
            - Age: {age}
            - Sex: {sex_at_birth}
            - Presenting Symptoms: {presenting_symptoms}
            - Duration of Symptoms: {duration_of_symptoms}
            - Past Medical History: {past_medical_history}
            - Current Medications: {current_medications}
            - Relevant Social History: {relevant_social_history}
            - Physical Examination Findings: {physical_examination_findings}
            - Any relevant Laboratory or Imaging results: {lab_or_imaging_results}
            """
            
            
            if st.button("Generate Differential Diagnosis"):
                # Your differential diagnosis generation code goes here
                if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                    ddx_output_text = answer_using_prefix_openai(ddx_prefix, ddx_sample_question, ddx_sample_answer, ddx_prompt, temperature=0.3, history_context='')
                    
                else:
                    ddx_output_text = answer_using_prefix(ddx_prefix, ddx_sample_question, ddx_sample_answer, ddx_prompt, temperature=0.3, history_context='')
                # st.write("Differential Diagnosis will appear here...")
                
                ddx_download_str = []
                
                with st.expander("Differential Diagnosis Draft", expanded=False):
                    st.info(f'Topic: {ddx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {ddx_output_text}', icon="🤖")                         
                    ddx_download_str = f"{disclaimer}\n\nDifferential Diagnoses for {ddx_prompt}:\n\n{ddx_output_text}"
                    if ddx_download_str:
                        st.download_button('Download', ddx_download_str, key = 'alt_dx_questions')
                        
                        
        # Alternative Diagnosis Generator
        if ddx_strategy == "Find Alternative Diagnoses to Consider":
            # st.subheader("Alternative Diagnosis Generator")
            
            alt_dx_prompt = st.text_input("Enter your presumed diagnosis.")

            if st.button("Generate Alternative Diagnoses"):
                if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                    alt_dx_output_text = answer_using_prefix_openai(alt_dx_prefix, alt_dx_sample_question, alt_dx_sample_answer, alt_dx_prompt, temperature=0.0, history_context='')
                    
                else:
                    alt_dx_output_text = answer_using_prefix(alt_dx_prefix, alt_dx_sample_question, alt_dx_sample_answer, alt_dx_prompt, temperature=0.0, history_context='')
                if st.session_state.model == "google/palm-2-chat-bison":
                    st.write("Alternative Diagnoses:", alt_dx_output_text)
                alt_dx_download_str = []
                with st.expander("Alternative Diagnoses Draft", expanded=False):
                    st.info(f'Topic: {alt_dx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {alt_dx_output_text}', icon="🤖")
                    alt_dx_download_str = f"{disclaimer}\n\nAlternative Diagnoses for {alt_dx_prompt}:\n\n{alt_dx_output_text}"
                    if alt_dx_download_str:
                        st.download_button('Download', alt_dx_download_str, key = 'alt_dx_questions')

    with tab3:

        pt_ed_health_literacy = st.radio("Pick a desired health literacy level:", ("General Public Medical Knowlege", "Advanced Medical Knowledge"))
        
        
        
        if pt_ed_health_literacy == "General Public Medical Knowlege":
            pt_ed_content_sample = pt_ed_basic_example

        if pt_ed_health_literacy == "Intermediate":
            pt_ed_content_sample = pt_ed_intermediate_example
        if pt_ed_health_literacy == "Advanced Medical Knowledge":
            pt_ed_content_sample = pt_ed_advanced_example
        
        sample_topic = "dietary guidance for a patient with diabetes, kidney disease, hypertension, obesity, and CAD"
        patient_ed_temp = st.session_state.temp
        my_ask_for_pt_ed = st.text_area("Generate patient education materials:", placeholder="e.g., dietary guidance needed for obesity", label_visibility='visible', height=100)
        my_ask_for_pt_ed = "Generate patient education materials for: " + my_ask_for_pt_ed.replace("\n", " ")
        my_ask_for_pt_ed = my_ask_for_pt_ed + "with health literacy level: " + pt_ed_health_literacy
        if st.button("Click to Generate **Draft** Custom Patient Education Materials"):
            st.info("Review all content carefully before considering any use!")
            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                pt_ed_output_text = answer_using_prefix_openai(pt_ed_system_content, sample_topic, pt_ed_content_sample, my_ask_for_pt_ed, patient_ed_temp, history_context="")
                
            else:
                pt_ed_output_text = answer_using_prefix(pt_ed_system_content, sample_topic, pt_ed_content_sample, my_ask_for_pt_ed, patient_ed_temp, history_context="")
            if st.session_state.model == "google/palm-2-chat-bison":
                st.write("Patient Education:", pt_ed_output_text)

            
            pt_ed_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Patient Education Draft", expanded=False):
                st.info(f'Topic: {my_ask_for_pt_ed}',icon="🧐")
                st.success(f'Draft Patient Education Materials: **REVIEW CAREFULLY FOR ERRORS** \n\n {pt_ed_output_text}', icon="🤖")      
                pt_ed_download_str = f"{disclaimer}\n\nDraft Patient Education Materials: {my_ask_for_pt_ed}:\n\n{pt_ed_output_text}"
                if pt_ed_download_str:
                        st.download_button('Download', pt_ed_download_str, key = 'pt_ed_questions')
                        
          
    
    with tab5:

        st.warning("This is just skimming the internet for medical answers even with the `deep` option. It is NOT reliable nor is it a replacement for a full medical reference. More development to come.")
        search_temp = st.session_state.temp
        deep = st.checkbox("Deep search (even more experimental)", value=False)
        if deep:
            max = st.slider("Max number of sites to analyze deeply", 1, 5, 1)
        domain = "Analyze only reputable sites."
                 
        set_domain = st.selectbox("Select a domain to emphasize:", ( "Medscape.com", "CDC.gov", "You specify a domain", "Any", ))
        if set_domain == "UpToDate.com":
            domain = "site: UpToDate.com, "
        if set_domain == "CDC.gov":
            domain = "site: cdc.gov, "
        if set_domain == "Medscape.com":
            domain = "site: medscape.com, "
        if set_domain == "PubMed":
            domain = "site: pubmed.ncbi.nlm.nih.gov "
        if set_domain == "Google Scholar":
            domain = "site: scholar.google.com "
        if set_domain == "Any":
            domain = "only use reputable sites, "
        if set_domain == "You specify a domain":
            domain = "site: " + st.text_input("Enter a domain to emphasize:", placeholder="e.g., cdc.gov, pubmed.ncbi.nlm.nih.gov, etc.", label_visibility='visible') + ", "
        
        my_ask_for_websearch = st.text_area("Skim the web to answer your question:", placeholder="e.g., how can I prevent kidney stones, what is the weather in Chicago tomorrow, etc.", label_visibility='visible', height=100)
        my_ask_for_websearch = domain + my_ask_for_websearch.replace("\n", " ")

        
        if st.button("Enter your question for a fun (NOT authoritative) draft websearch tool"):
            st.info("Review all content carefully before considering any use!")
            raw_output, urls = websearch(my_ask_for_websearch, deep, max)
            
            if not deep:
                # raw_output = json.dumps(raw_output)
                new_text = ""
                i = 0
                for item in raw_output["data"]:
                    snippet = item["snippet"]
                    domain = item["domain"]
                    new_text += f"Snippet: {snippet} from domain: {domain}\n\n"
                raw_output = new_text
                with st.expander("Content Reviewed", expanded=False):
                    st.write(raw_output)
                    # st.write(urls)
            else:
                raw_output = truncate_text(raw_output, 1000)
                with st.expander("Content reviewed", expanded=False):
                # raw_output = limit_tokens(raw_output, 8000)
                    st.write(f'Truncated at 1000 tokens: \n\n  {raw_output}')
            my_ask_for_websearch = f'User: {my_ask_for_websearch} \n\n Content: {raw_output}'
            if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                st.write("Your answer from sifting the web:")
                skim_output_text = answer_using_prefix_openai(interpret_search_results_prefix, "", '', my_ask_for_websearch, search_temp, history_context="")
                
            else:
                st.write("Your answer from sifting the web:")
                skim_output_text = answer_using_prefix(interpret_search_results_prefix, "", '', my_ask_for_websearch, search_temp, history_context="")
            if st.session_state.model == "google/palm-2-chat-bison":
                st.write("Answer:", skim_output_text)

            
            skim_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Links Identified"):
                for item in urls:
                    st.write(item)
            with st.expander("Sifting Web Summary", expanded=False):
                st.info(f'Topic: {my_ask_for_websearch}',icon="🧐")
                st.success(f'Your Sifted Response: **REVIEW CAREFULLY FOR ERRORS** \n\n {skim_output_text}', icon="🤖")      
                skim_download_str = f"{disclaimer}\n\nSifted Summary: {my_ask_for_websearch}:\n\n{skim_output_text}"
                if skim_download_str:
                        st.download_button('Download', skim_download_str, key = 'skim_questions')
        
    
    
    
    
    with tab6:
        
        if "pdf_user_question" not in st.session_state:
            st.session_state["pdf_user_question"] = []
        if "pdf_user_answer" not in st.session_state:
            st.session_state["pdf_user_answer"] = []
            
            
        # st.sidebar.title("Settings and Preliminary Outputs")
        # num_eval_questions =st.number_input("Specify how many questions you'd like to generate (then press enter on your keyboard):", min_value=0, max_value=10, value=0, step=1)
        num_eval_questions = 0
        
        embedding_option = "OpenAI Embeddings"

        
        retriever_type = "SIMILARITY SEARCH"

        # Use RecursiveCharacterTextSplitter as the default and only text splitter
        splitter_type = "RecursiveCharacterTextSplitter"

        uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                        "pdf", "txt"], accept_multiple_files=True)



        if uploaded_files is not None:
            # st.write("Yes, we have the file.")
            # Check if Uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
            # if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

            # Load and process the uploaded PDF or TXT files.
            loaded_text = load_docs(st.session_state.last_uploaded_files)
            # st.write("Documents uploaded and 'read.'")

            # Split the document into chunks
            splits = split_texts(loaded_text, chunk_size=1250,
                                overlap=200, split_method=splitter_type)

            # Display the number of text chunks
            num_chunks = len(splits)
            # st.write(f"Number of text chunks: {num_chunks}")

            # Embed using OpenAI embeddings
                # Embed using OpenAI embeddings or HuggingFace embeddings

            embeddings = OpenAIEmbeddings()
            


            st.session_state.pdf_retriever = create_retriever(embeddings, splits, retriever_type)


            # Initialize the RetrievalQA chain with streaming output
            callback_handler = StreamingStdOutCallbackHandler()
            callback_manager = CallbackManager([callback_handler])

            chat_openai = ChatOpenAI(
                streaming=False, callback_manager=callback_manager, verbose=True, temperature=0.3)
            
            
            _qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=st.session_state.pdf_retriever, chain_type="stuff", verbose=False)

            
        

            # Check if there are no generated question-answer pairs in the session state
            
            # if 'eval_set' not in st.session_state:
            #     # Use the generate_eval function to generate question-answer pairs
            #     # num_eval_questions = 10  # Number of question-answer pairs to generate
            #     st.session_state.eval_set = generate_eval(
            #         loaded_text, num_eval_questions, 3000)

            #     # Display the question-answer pairs in the sidebar with smaller text
            # for i, qa_pair in enumerate(st.session_state.eval_set):
            #     st.sidebar.markdown(
            #         f"""
            #         <div class="css-card">
            #         <span class="card-tag">Question {i + 1}</span>
            #             <p style="font-size: 12px;">{qa_pair['question']}</p>
            #             <p style="font-size: 12px;">{qa_pair['answer']}</p>
            #         </div>
            #         """,
            #         unsafe_allow_html=True,
            #     )
                # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
                # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
            st.write("Ready to answer your questions!")

                # Question and answering
    
            pdf_chat_option = st.radio("Select an Option", ("Summary", "Custom Question"))
            if pdf_chat_option == "Summary":
                user_question = "Summary: Using context provided, generate a concise and comprehensive summary. Key Points: Generate a list of Key Points by using a conclusion section if present and the full context otherwise."
            if pdf_chat_option == "Custom Question":
                user_question = st.text_input("Please enter your own question about the PDF(s):")
                user_question = "Using context provided, answer the user question: " + user_question
            
            if st.button("Generate a Response"):
                index_context = f'Use only the reference document for knowledge. Question: {user_question}'
                pdf_answer = fn_qa_run(_qa, index_context)
                st.session_state.pdf_user_question.append(user_question)  
                st.session_state.pdf_user_answer.append(pdf_answer)  
                # st.write("Answer:", answer)
                pdf_chat_download_str = []
                with st.expander("PDF Questions", expanded=False):                     
                    st.info(f'Your Question: {user_question}',icon="🧐")
                    st.success(f'PDF Response:\n\n {pdf_answer}', icon="🤖")      
                    pdf_download_str = f"{disclaimer}\n\nPDF Answers: {user_question}:\n\n{pdf_answer}"
                    if pdf_download_str:
                            st.download_button('Download', pdf_download_str, key = 'pdf_questions')

        
                    
