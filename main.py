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


    with st.expander('About GPT and Med Ed - Important Disclaimer'):
        st.write("Author: David Liebovitz, MD, Northwestern University")
        st.info(disclaimer)
        st.session_state.model = st.radio("Select model (Costs: $, $$, and $$$$)", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"), index=0)
        st.session_state.temp = st.slider("Select temperature (Higher values more creative but tangential and more error prone)", 0.0, 1.0, 0.3, 0.01)
        st.write("Last updated 8/12/23")



    tab1, tab2, tab3, tab4 = st.tabs(["Learn", "Draft Communication", "Patient Education", "Differential Diagnosis",])

        
    with tab1:
    
        persona = st.radio("Select teaching persona", ("Teacher 1 (academic)", "Teacher 2 (analogies)", "Create Your Own"), index=0)

        if persona == "Create Your Own":
            system_context = st.sidebar.text_area('Enter a persona description: (e.g., "You are a cardiologist who is teaching a medical student on inpatient service.")', 
                                                  placeholder="e.g, you are a medical educator skilled in educational techniques", label_visibility='visible', height=100, key="system_context")
            system_context = system_context.replace("\n", " ")
            system_context = system_context + " " + base_teacher
            if st.sidebar.button("Set Persona"):
                st.sidebar.info("Your persona is set.")
        elif persona == "Teacher 1 (academic)":
            system_context = teacher1
        elif persona == "Teacher 2 (analogies)":
            system_context = teacher2
        
        show_prompt = st.checkbox("Show selected persona details")
        if show_prompt:
            st.sidebar.markdown(system_context)
            
        my_ask = st.text_area('Enter a topic: (e.g., RAAS, Frank-Starling, sarcoidosis, etc.)',placeholder="e.g., sarcoidosis", label_visibility='visible', height=100, key="my_ask")
        my_ask = my_ask.replace("\n", " ")
        my_ask = "Teach me about: " + my_ask
        
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
        # st.subheader("Patient Communication")
        col1, col2 = st.columns(2)
        with col2:
            health_literacy_level = st.radio("Output optimized for:", ("Low Health Literacy", "High Health Literacy"))
   

        with col1:
            task = st.radio("What do you want to do?", ("Generate discharge instructions", "Annotate a patient result"))

        if task == "Generate discharge instructions":
            answer = ''
            start_time = time.time()
            surg_procedure = st.text_area("Please enter the procedure performed and any special concerns.", placeholder="e.g., right total hip arthroplasty", label_visibility='visible',)
            dc_meds = st.text_area("Please enter the discharge medications.", placeholder="e.g., lisinopril 10 mg daily for HTN", label_visibility='visible',)
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
                alt_dx_output_text = answer_using_prefix(alt_dx_prefix, alt_dx_sample_question, alt_dx_sample_answer, alt_dx_prompt, temperature=0.0, history_context='')
                alt_dx_download_str = []
                with st.expander("Alternative Diagnoses Draft", expanded=False):
                    st.info(f'Topic: {alt_dx_prompt}',icon="🧐")
                    st.success(f'Educational Use Only: **NOT REVIEWED FOR CLINICAL CARE** \n\n {alt_dx_output_text}', icon="🤖")
                    alt_dx_download_str = f"{disclaimer}\n\nAlternative Diagnoses for {alt_dx_prompt}:\n\n{alt_dx_output_text}"
                    if alt_dx_download_str:
                        st.download_button('Download', alt_dx_download_str, key = 'alt_dx_questions')

    with tab3:

        pt_ed_health_literacy = st.radio("Pick a desired health literacy level:", ("Basic", "Intermediate", "Advanced"))
        
        
        
        if pt_ed_health_literacy == "Basic":
            pt_ed_content_sample = pt_ed_basic_example

        if pt_ed_health_literacy == "Intermediate":
            pt_ed_content_sample = pt_ed_intermediate_example
        if pt_ed_health_literacy == "Advanced":
            pt_ed_content_sample = pt_ed_advanced_example
        
        sample_topic = "dietary guidance for a patient with diabetes, kidney disease, hypertension, obesity, and CAD"
        patient_ed_temp = st.session_state.temp
        my_ask_for_pt_ed = st.text_area("Generate patient education materials:", placeholder="e.g., dietary guidance needed for obesity", label_visibility='visible', height=100)
        if st.button("Click to Generate **Draft** Custom Patient Education Materials"):
            st.info("Review all content carefully before considering any use!")
            pt_ed_output_text = answer_using_prefix(pt_ed_system_content, sample_topic, pt_ed_content_sample, my_ask_for_pt_ed, patient_ed_temp, history_context="")

            
            pt_ed_download_str = []
            
            # ENTITY_MEMORY_CONVERSATION_TEMPLATE
            # Display the conversation history using an expander, and allow the user to download it
            with st.expander("Patient Education Draft", expanded=False):
                st.info(f'Topic: {my_ask_for_pt_ed}',icon="🧐")
                st.success(f'Draft Patient Education Materials: **REVIEW CAREFULLY FOR ERRORS** \n\n {pt_ed_output_text}', icon="🤖")      
                pt_ed_download_str = f"{disclaimer}\n\nDraft Patient Education Materials: {my_ask_for_pt_ed}:\n\n{pt_ed_output_text}"
                if pt_ed_download_str:
                        st.download_button('Download', pt_ed_download_str, key = 'pt_ed_questions')