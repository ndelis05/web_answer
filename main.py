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
import os
import fitz
from io import StringIO

@st.cache_data
def display_articles_with_streamlit(articles):
    i = 1
    for article in articles:
        st.write(f"{i}. {article['title']}[{article['year']}]({article['link']})")
        i+=1
        # st.write("---")  # Adds a horizontal line for separation

def set_llm_chat(model, temperature):
    if model == "openai/gpt-3.5-turbo":
        model = "gpt-3.5-turbo"
    if model == "openai/gpt-3.5-turbo-16k":
        model = "gpt-3.5-turbo-16k"
    if model == "openai/gpt-4":
        model = "gpt-4"
    if model == "gpt-4" or model == "gpt-3.5-turbo" or model == "gpt-3.5-turbo-16k":
        return ChatOpenAI(model=model, openai_api_base = "https://api.openai.com/v1/", openai_api_key = st.secrets["OPENAI_API_KEY"], temperature=temperature)
    else:
        headers={ "HTTP-Referer": "https://fsm-gpt-med-ed.streamlit.app", # To identify your app
          "X-Title": "GPT and Med Ed"}
        return ChatOpenAI(model = model, openai_api_base = "https://openrouter.ai/api/v1", openai_api_key = st.secrets["OPENROUTER_API_KEY"], temperature=temperature, max_tokens = 500, headers=headers)
def truncate_text(text, max_characters):
    if len(text) <= max_characters:
        return text
    else:
        truncated_text = text[:max_characters]
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
    # st.write(url_list)
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

@st.cache_data
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

@st.cache_data
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
    # def display_search_results(json_data):
    #     data = json_data['data']
    #     for item in data:
    #         st.sidebar.markdown(f"### [{item['title']}]({item['url']})")
    #         st.sidebar.write(item['snippet'])
    #         st.sidebar.write("---")
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

@st.cache_data
def pubmed_abstracts(search_terms, search_type="all"):
    # URL encoding
    search_terms_encoded = requests.utils.quote(search_terms)

    # Define the publication type filter based on the search_type parameter
    if search_type == "all":
        publication_type_filter = ""
    elif search_type == "clinical trials":
        publication_type_filter = "+AND+Clinical+Trial[Publication+Type]"
    elif search_type == "reviews":
        publication_type_filter = "+AND+Review[Publication+Type]"
    else:
        raise ValueError("Invalid search_type parameter. Use 'all', 'clinical trials', or 'reviews'.")

    # Construct the search query with the publication type filter
    search_query = f"{search_terms_encoded}{publication_type_filter}"
    
    # Query to get the top 20 results
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_query}&retmode=json&retmax=20&api_key={st.secrets['pubmed_api_key']}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        # Check if no results were returned, and if so, use a longer approach
        if 'count' in data['esearchresult'] and int(data['esearchresult']['count']) == 0:
            return st.write("No results found. Try a different search or try again after re-loading the page.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching search results: {e}")
        return []

    ids = data['esearchresult']['idlist']
    articles = []

    for id in ids:
        details_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={id}&retmode=json&api_key={st.secrets['pubmed_api_key']}"
        try:
            details_response = requests.get(details_url)
            details_response.raise_for_status()  # Raise an exception for HTTP errors
            details = details_response.json()
            if 'result' in details and str(id) in details['result']:
                article = details['result'][str(id)]
                year = article['pubdate'].split(" ")[0]
                if year.isdigit():
                    articles.append({
                        'title': article['title'],
                        'year': year,
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{id}"
                    })
            else:
                st.warning(f"Details not available for ID {id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching details for ID {id}: {e}")
        time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    # Second query: Get the abstract texts for the top 10 results
    abstracts = []
    for id in ids:
        abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id}&retmode=text&rettype=abstract&api_key={st.secrets['pubmed_api_key']}"
        try:
            abstract_response = requests.get(abstract_url)
            abstract_response.raise_for_status()  # Raise an exception for HTTP errors
            abstract_text = abstract_response.text
            if "API rate limit exceeded" not in abstract_text:
                abstracts.append(abstract_text)
            else:
                st.warning(f"Rate limit exceeded when fetching abstract for ID {id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching abstract for ID {id}: {e}")
        time.sleep(1)  # Introduce a delay to avoid hitting rate limits only if there's an error

    return articles, "\n".join(abstracts)

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
    stream = True
    if st.session_state.model == "anthropic/claude-instant-v1":
        stream = False
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
    max_tokens = 750,
    stream = stream,   
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

@st.cache_data  # Updated decorator name from cache_data to cache
def load_docs(files):
    all_text = ""
    for file in files:
        file_extension = os.path.splitext(file.name)[1]
        if file_extension == ".pdf":
            pdf_data = file.read()  # Read the file into bytes
            pdf_reader = fitz.open("pdf", pdf_data)  # Open the PDF from bytes
            text = ""
            for page in pdf_reader:
                text += page.get_text()
            all_text += text

        elif file_extension == ".txt":
            stringio = StringIO(file.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text += text
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text


@st.cache_data
def create_retriever(texts):  
    
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",
                                  openai_api_base = "https://api.openai.com/v1/",
                                  openai_api_key = st.secrets['OPENAI_API_KEY']
                                  )
    try:
        vectorstore = FAISS.from_texts(texts, embeddings)
    except (IndexError, ValueError) as e:
        st.error(f"Error creating vectorstore: {e}")
        return
    retriever = vectorstore.as_retriever(k=5)

    return retriever

@st.cache_data
def split_texts(text, chunk_size, overlap, split_method):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)

    splits = text_splitter.split_text(text)
    if not splits:
        # st.error("Failed to split document")
        st.stop()

    return splits

@st.cache_data
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



@st.cache_data
def prepare_rag(text):
    splits = split_texts(text, chunk_size=1000, overlap=100, split_method="recursive")
    st.session_state.retriever = create_retriever(splits)
    llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.retriever)
    return rag
    
    

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
    
if "pdf_user_question" not in st.session_state:
    st.session_state["pdf_user_question"] = []
if "pdf_user_answer" not in st.session_state:
    st.session_state["pdf_user_answer"] = []

if "last_uploaded_files" not in st.session_state:
    st.session_state["last_uploaded_files"] = []
    
if "abstract_questions" not in st.session_state:
    st.session_state["abstract_questions"] = []
    
if "abstract_answers" not in st.session_state:
    st.session_state["abstract_answers"] = []

if "last_answer" not in st.session_state:
    st.session_state["last_answer"] = []

if "abstracts" not in st.session_state:
    st.session_state["abstracts"] = ""
    
if "your_question" not in st.session_state:
    st.session_state["your_question"] = ""
    
if "texts" not in st.session_state:
    st.session_state["texts"] = ""
    
if "citations" not in st.session_state:
    st.session_state["citations"] = ""
    
if "search_terms" not in st.session_state:
    st.session_state["search_terms"] = ""   

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
        st.session_state.model = st.selectbox("Model Options", ("openai/gpt-3.5-turbo", "openai/gpt-3.5-turbo-16k",  "openai/gpt-4", "anthropic/claude-instant-v1", "google/palm-2-chat-bison", "meta-llama/codellama-34b-instruct", "meta-llama/llama-2-70b-chat", "gryphe/mythomax-L2-13b", "nousresearch/nous-hermes-llama2-13b"), index=1)
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
        if st.session_state.model == "gryphe/mythomax-L2-13b":
            st.markdown("[Information on Gryphe's Mythomax](https://huggingface.co/Gryphe/MythoMax-L2-13b)")
        if st.session_state.model == "meta-llama/codellama-34b-instruct":
            st.markdown("[Information on Meta's CodeLlama](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)")
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
            
        # if st.button("Clear Memory (when you don't want to send prior context)"):
        #     st.session_state.history = []
        #     st.session_state.output_history = []
        #     clear_session_state_except_password_correct()
        #     st.write("Memory cleared")
        
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

        st.warning("This is just skimming the internet for medical answers even with the `deep` option. It is NOT reliable nor is it a replacement for a full medical reference. Errors arise when websites restrict automated data retrieval.")
        if 'temp' not in st.session_state:
            st.session_state.temp = 0.3
        search_temp = st.session_state.temp

        domain = "Analyze only reputable sites."
                 
        set_domain = st.selectbox("Select a domain to emphasize:", ("NLM Bookshelf", "Ask PubMed", "Medscape", "CDC", "Stat Pearls", "You specify a domain", "Any", ))
        if set_domain == "UpToDate.com":
            domain = "site: UpToDate.com, "
        if set_domain == "CDC":
            domain = "site: cdc.gov, "
        if set_domain == "Medscape":
            domain = "site: medscape.com, "
        if set_domain == "NLM Bookshelf":
            domain = "site: ncbi.nlm.nih.gov/books/, "
        if set_domain == "Stat Pearls":
            domain = "site: statpearls.com, "

        if set_domain == "Google Scholar":
            domain = "site: scholar.google.com, "
        if set_domain == "Any":
            domain = "only use reputable sites, "
        if set_domain == "You specify a domain":
            domain = "site: " + st.text_input("Enter a full web domain to emphasize:", placeholder="e.g., cdc.gov, pubmed.ncbi.nlm.nih.gov, etc.", label_visibility='visible') + ", "
        
        if set_domain != "Ask PubMed":
            deep = st.checkbox("Deep search (Retrieves full text from a few pages instead of snippets from many pages)", value=False)
            if deep:
                max = st.slider("Max number of sites to analyze deeply", 1, 5, 1)
        
        if set_domain == "Ask PubMed":
            st.warning("""This PubMed option isn't web scraping like the others. Here, we perform a PubMed search and then search a vector database of retrieved abstracts in order to answer your question. Clearly,
            this will often be inadequate and is intended to illustrate an AI approach that will become (much) better with time. View citations and retrieved abtracts on the left sidebar. """)

            search_type = st.radio("Select an Option", ("all", "clinical trials", "reviews"), horizontal=True)
            your_question = st.text_input("Your question for PubMed", placeholder="Enter your question here")
            st.session_state.your_question = your_question
            
            if st.session_state.your_question != "":
                search_terms = answer_using_prefix("Convert the user's question into relevant PubMed search terms; include related MeSH terms to improve sensitivity.", 
                                            "What are the effects of intermittent fasting on weight loss in adults over 50?",
                                            "(Intermittent fasting OR Fasting[MeSH]) AND (Weight loss OR Weight Reduction Programs[MeSH]) AND (Adults OR Middle Aged[MeSH]) AND Age 50+ ", st.session_state.your_question, 0.5, None)
                                            
                # st.write(f'Here are your search terms: {search_terms}')                                   
                st.session_state.search_terms = search_terms
                with st.sidebar.expander("Current Question", expanded=False):
                    st.write(st.session_state.your_question)
                    st.write('Search terms used: ' + st.session_state.search_terms)
                
                if st.session_state.search_terms != "":
                    try:
                        with st.spinner("Searching PubMed... (Temperamental - ignore errors if otherwise working. NLM throttles queries; API access can take a minute or two.)"):
                            st.session_state.citations, st.session_state.abstracts = pubmed_abstracts(st.session_state.search_terms, search_type=search_type)
                    except:
                        st.warning("Insufficient findings to parse results.")
                        st.stop()
                    if st.session_state.citations == [] or st.session_state.abstracts == "":
                        st.warning("The PubMed API is tempermental. Refresh and try again.")
                        st.stop()


                # st.write(st.session_state.citations)
                # st.write(st.session_state.abstracts)

            with st.sidebar.expander("Show citations"):
                display_articles_with_streamlit(st.session_state.citations)
            with st.sidebar.expander("Show abstracts"):
                st.write(st.session_state.abstracts)
            system_context_abstracts = """You receive user query terms and PubMed abstracts for those terms as  your inputs. You first provide a composite summary of all the abstracts emphasizing any of their conclusions. Next,
            you provide key points from the abstracts in order address the user's likely question based on the on the query terms.       
            """

            # Unblock below if you'd like to submit the full abtracts. This is not recommended as it is likely to be too long for the model.

            # prompt_for_abstracts = f'User question: {your_question} Abstracts: {st.session_state.abstracts} /n/n Generate one summary covering all the abstracts and then list key points to address likely user questions.'

            # with st.spinner("Waiting on LLM analysis of abstracts..."):
            #     full_answer = answer_using_prefix(system_context_abstracts, "","",prompt_for_abstracts, 0.5, None, st.session_state.model)
            # with st.expander("Show summary and key points"):
            #     st.write(f'Here is the full abstracts inferred answers: {full_answer}')


            # st.write("'Reading' all the abstracts to answer your question. This may take a few minutes.")


            st.info("""Next, words in the abstracts are converted to numbers for analysis. This is called embedding and is performed using an OpenAI [embedding model](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and then indexed for searching. Lastly,
                    your selected model (e.g., gpt-3.5-turbo-16k) is used to answer your question.""")


            if st.session_state.abstracts != "":
                
                # with st.spinner("Embedding text from the abstracts (lots of math can take a couple minutes to change words into vectors)..."):
                #     st.session_state.retriever = create_retriever(st.session_state.abstracts)

                with st.spinner("Splitting text from the abstracts into concept chunks..."):
                    st.session_state.texts = split_texts(st.session_state.abstracts, chunk_size=1250,
                                                overlap=200, split_method="splitter_type")
                with st.spinner("Embedding the text (converting words to vectors) and indexing to answer questions about the abtracts (Takes a couple minutes)."):
                    st.session_state.retriever = create_retriever(st.session_state.texts)


                # openai.api_base = "https://openrouter.ai/api/v1"
                # openai.api_key = st.secrets["OPENROUTER_API_KEY"]

                llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
                # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_base = "https://api.openai.com/v1/")

                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.retriever)

            else:
                st.warning("No files uploaded.")       
                st.write("Ready to answer your questions!")



            user_question_abstract = st.text_input("Ask followup questions about the retrieved abstracts. Here was your initial question:", your_question)

            index_context = f'Use only the reference document for knowledge. Question: {user_question_abstract}'
            if st.session_state.abstracts != "":
                with st.spinner("Reviewing the abstracts to formulate a response..."):
                    abstract_answer = qa(index_context)

                if st.button("Ask more about the abstracts"):
                    index_context = f'Use only the reference document for knowledge. Question: {user_question_abstract}'
                    with st.spinner("Reviewing the abstracts to formulate a response..."):
                        abstract_answer = qa(index_context)

                # Append the user question and PDF answer to the session state lists
                st.session_state.abstract_questions.append(user_question_abstract)
                st.session_state.abstract_answers.append(abstract_answer)

                # Display the PubMed answer
                st.write(abstract_answer["result"])

                # Prepare the download string for the PDF questions
                abstract_download_str = f"{disclaimer}\n\nPDF Questions and Answers:\n\n"
                for i in range(len(st.session_state.abstract_questions)):
                    abstract_download_str += f"Question: {st.session_state.abstract_questions[i]}\n"
                    abstract_download_str += f"Answer: {st.session_state.abstract_answers[i]['result']}\n\n"

                # Display the expander section with the full thread of questions and answers
                with st.expander("Your Conversation with your Abstracts", expanded=False):
                    for i in range(len(st.session_state.abstract_questions)):
                        st.info(f"Question: {st.session_state.abstract_questions[i]}", icon="🧐")
                        st.success(f"Answer: {st.session_state.abstract_answers[i]['result']}", icon="🤖")

                    if abstract_download_str:
                        st.download_button('Download', abstract_download_str, key='abstract_questions_downloads')
            
    
        else:
            my_ask_for_websearch = st.text_area("Skim the web to answer your question:", placeholder="e.g., how can I prevent kidney stones, what is the weather in Chicago tomorrow, etc.", label_visibility='visible', height=100)
            my_ask_for_websearch_part1 = domain + my_ask_for_websearch.replace("\n", " ")

            
            if st.button("Enter your question for a fun (NOT authoritative) draft websearch tool"):
                st.info("Review all content carefully before considering any use!")
                raw_output, urls = websearch(my_ask_for_websearch_part1, deep, max)
                
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
                    my_ask_for_websearch = f'User: {my_ask_for_websearch} \n\n Content basis for your answer: {raw_output}'

                
                    if st.session_state.model == "openai/gpt-3.5-turbo" or st.session_state.model == "openai/gpt-3.5-turbo-16k" or st.session_state.model == "openai/gpt-4":
                        st.warning("Be sure to validate! This just used web snippets to answer your question!")
                        skim_output_text = answer_using_prefix_openai(interpret_search_results_prefix, "", '', my_ask_for_websearch, search_temp, history_context="")
                        
                    else:
                        st.warning("Be sure to validate! This just used web snippets to answer your question!")
                        skim_output_text = answer_using_prefix(interpret_search_results_prefix, "", '', my_ask_for_websearch, search_temp, history_context="")
                    if st.session_state.model == "google/palm-2-chat-bison":
                        st.write("Answer:", skim_output_text)
                        
                else:
                    # raw_output = truncate_text(raw_output, 5000)
                    with st.spinner('Searching the web and converting findings to vectors...'):
                        rag = prepare_rag(raw_output)                
                    with st.expander("Content reviewed", expanded=False):
                        raw_output = truncate_text(raw_output, 25000)
                        st.write(f'Truncated below at ~5000 tokens, but all in vector database:  \n\n  {raw_output}')
                    with st.spinner('Searching the vector database to assemble your answer...'):
                        skim_output_text = rag(my_ask_for_websearch)
                    skim_output_text = skim_output_text["result"]
                    st.warning(f'Be sure to validate! This just used {max} webpage(s) to answer your question!')
                    st.write(skim_output_text)

                
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
        st.header("Chat with your PDFs!")
        st.info("""Embeddings, i.e., reading your file(s) and converting words to numbers, are created using an OpenAI [embedding model](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) and indexed for searching. Then,
                your selected model (e.g., gpt-3.5-turbo-16k) is used to answer your questions.""")
        st.warning("""Some PDFs are images and not formatted text. If the summary feature doesn't work, you may first need to convert your PDF
                   using Adobe Acrobat. Choose: `Scan and OCR`,`Enhance scanned file` \n   Alternatively, sometimes PDFs are created with 
                   unusual fonts or LaTeX symbols. Export the file to Word, re-save as a PDF and try again. Save your updates, upload and voilà, you can chat with your PDF! """)
        uploaded_files = []
        # os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

        uploaded_files = st.file_uploader("Choose your file(s)", accept_multiple_files=True)

        if uploaded_files is not None:
            documents = load_docs(uploaded_files)
            texts = split_texts(documents, chunk_size=1250,
                                        overlap=200, split_method="splitter_type")

            retriever = create_retriever(texts)

            # openai.api_base = "https://openrouter.ai/api/v1"
            # openai.api_key = st.secrets["OPENROUTER_API_KEY"]

            llm = set_llm_chat(model=st.session_state.model, temperature=st.session_state.temp)
            # llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_base = "https://api.openai.com/v1/")

            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        else:
            st.warning("No files uploaded.")       
            st.write("Ready to answer your questions!")

        col1, col2 = st.columns(2)
        with col1:
            pdf_chat_option = st.radio("Select an Option", ("Summary", "Custom Question", "Generate MCQs", "Appraise a Clinical Trial"))
        if pdf_chat_option == "Summary":
            with col2:
                summary_method= st.radio("Select a Summary Method", ("Standard Summary", "Chain of Density"))
            word_count = st.slider("Approximate Word Count for the Summary. Most helpful for very long articles", 100, 1000, 250)
            if summary_method == "Chain of Density":
                st.write("Generated with [Chain of Density](https://arxiv.org/abs/2309.04269) methodology.")
                user_question = chain_of_density_summary_template
                user_question = user_question.format(word_count=word_count, context = "{context}")
            if summary_method == "Standard Summary":
                user_question = key_points_summary_template
                user_question = user_question.format(word_count=word_count, context = "{context}")
            
        if pdf_chat_option == "Custom Question":
            user_question = st.text_input("Please enter your own question about the PDF(s):")
            
        if pdf_chat_option == "Generate MCQs":
            num_mcq = st.slider("Number of MCQs", 1, 10, 3)
            with col2: 
                mcq_options = st.radio("Select a Sub_Option", ("Generate MCQs", "Generate MCQs on a Specific Topic"))
            
            if mcq_options == "Generate MCQs":
                user_question = mcq_generation_template
                user_question = user_question.format(num_mcq=num_mcq, context = "{context}")
                
            if mcq_options == "Generate MCQs on a Specific Topic":
                user_focus = st.text_input("Please enter a covered topic for the focus of your MCQ:")
                user_question = f'Topic for question generation: {user_focus}' + f'\n\n {mcq_generation_template}'
                user_question = user_question.format(num_mcq=num_mcq, context = "{context}")
        if pdf_chat_option == "Appraise a Clinical Trial":
            st.write('Note GPT4 is much better; may take a couple minutes to run.')
            # word_count = st.slider("Approximate Word Count for the Summary. Most helpful for very long articles", 100, 1000, 600)
            user_question = clinical_trial_template
            # user_question = user_question.format(word_count=word_count, context = "{context}")
        if st.button("Generate a Response"):
            # index_context = f'Use only the reference document for knowledge. Question: {user_question}'
            pdf_answer = qa(user_question)

            # Append the user question and PDF answer to the session state lists
            st.session_state.pdf_user_question.append(user_question)
            st.session_state.pdf_user_answer.append(pdf_answer)

            # Display the PDF answer
            st.write(pdf_answer["result"])

            # Prepare the download string for the PDF questions
            pdf_download_str = f"{disclaimer}\n\nPDF Questions and Answers:\n\n"
            for i in range(len(st.session_state.pdf_user_question)):
                pdf_download_str += f"Question: {st.session_state.pdf_user_question[i]}\n"
                pdf_download_str += f"Answer: {st.session_state.pdf_user_answer[i]['result']}\n\n"

            # Display the expander section with the full thread of questions and answers
            with st.expander("Your Conversation with your PDF", expanded=False):
                for i in range(len(st.session_state.pdf_user_question)):
                    st.info(f"Question: {st.session_state.pdf_user_question[i]}", icon="🧐")
                    st.success(f"Answer: {st.session_state.pdf_user_answer[i]['result']}", icon="🤖")

                if pdf_download_str:
                    st.download_button('Download', pdf_download_str, key='pdf_questions')
            

        
    
                    
