import streamlit as st
import openai
from prompts import *
import time
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
import requests
import json

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4"

if "current_response" not in st.session_state:
    st.session_state["current_response"] = ""
    
if "current_question" not in st.session_state:
    st.session_state["current_question"] = ""

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
def websearch(web_query: str, deep, scrape_method, max) -> float:
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
        if scrape_method != "Browserless":
            response_data = scrapeninja(urls, max)
        else:
            response_data = browserless(urls, max)
        # st.info("Web results reviewed.")
        return response_data, urls

    else:
        # st.info("Web snippets reviewed.")
        return response_data, urls
@st.cache_data
def browserless(url_list, max):
    # st.write(url_list)
    if max > 5:
        max = 5
    response_complete = []
    i = 0
    key = st.secrets["BROWSERLESS_API_KEY"]
    api_url = f'https://chrome.browserless.io/content?token={key}'
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json'
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
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
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
    # limited_text = limit_tokens(full_response, 12000)
    # st.write(f'Here is the lmited text: {limited_text}')
    return full_response

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
            # st.write(last_two_messages)
            
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
    
    proceed = st.checkbox("Acknowledge - this tool should be used for foundational knowledge only. Use responsibly. Do not use to learn about the latest treatments.")
    
    if proceed:
        
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
        
        learner = st.radio("Select your learner type", ("Medical Student", "Attending", "Advanced - extra dense and concise",))
        st.write("👇🏾Enter your question or topic at the 👇🏾bottom👇🏾 of the page. Thank you for using this tool responsibly.")            
        
        
                
        if prompt := st.chat_input("Enter your text for the dialog here!"):
            # Add user message to chat history
            prompt = prompt.format(learner=learner)
            st.session_state.current_question = prompt
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
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.current_response = full_response
                
                

        if st.session_state.current_response != "":  
            
            if st.checkbox("Check the current response for errors against the NLM Bookshelf Content"):
                st.markdown("Describe your area of concern. Be specific. The original response will be assessed for errors using content from the [NLM Bookshelf](https://www.ncbi.nlm.nih.gov/books/browse/).")
                topic_area_of_concern = st.text_input("Topic of concern: Be specific (as if doing a reference search) so the correct references are used from NLM.")
                if topic_area_of_concern != "":
                
                    raw_output, urls = websearch(topic_area_of_concern, True, "Browserless", 8)
                            
                    # raw_output = truncate_text(raw_output, 5000)
                    with st.spinner('Searching the web and converting findings to vectors...'):
                        rag = prepare_rag(raw_output)                
                    # with st.expander("Content reviewed", expanded=False):
                    #     raw_output = truncate_text(raw_output, 25000)
                    #     st.write(f'Truncated below at ~5000 tokens, but all in vector database:  \n\n  {raw_output}')
                    with st.spinner('Searching the vector database to assemble your answer...'):
                        skim_output_text = rag("Assess for errors of fact or omission in the following: " + st.session_state.current_response)
                    skim_output_text = skim_output_text["result"]
                    st.warning(f'This is using the NLM bookshelf to assess accuracy.')
                    

                    
                    # skim_download_str = []
                
                    # ENTITY_MEMORY_CONVERSATION_TEMPLATE
                    # Display the conversation history using an expander, and allow the user to download it
                    with st.expander("Sources Reviewed at NLM"):
                        for item in urls:
                            st.write(item)
                            
                    st.write(skim_output_text)
                    st.warning("You can see where you left off below. Click to reopen your last response and continue dialog.")
                    with st.expander("The Current Response Under Review"):
                        st.write(st.session_state.current_response)
                    # with st.expander("Sifting Web Summary", expanded=False):
                    #     st.info(f'Topic: {topic_area_of_concern}',icon="🧐")
                    #     st.success(f'Your Sifted Response: **REVIEW CAREFULLY FOR ERRORS** \n\n {skim_output_text}', icon="🤖")      
                    #     skim_download_str = f"{disclaimer}\n\nOriginal: {st.session_state.current_response}:\n\n{skim_output_text}"
                    #     if skim_download_str:
                    #             st.download_button('Download', skim_download_str, key = 'skim_questions')
                                
                                
            if st.sidebar.button("Clear Memory (when you want to switch topics)"):
                st.session_state.messages = []
                st.sidebar.write("Memory cleared")
                full_response = ""
                st.session_state.messages = [{'role': 'system', 'content': interactive_teacher},]
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.current_response = ""
            
