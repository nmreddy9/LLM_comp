import streamlit as st
from transformers import pipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

hfid = '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'

def summarize_text(input_text):

    repo_id = "pszemraj/long-t5-tglobal-base-16384-book-summary"

    llm = HuggingFaceHub(huggingfacehub_api_token=hfid, repo_id=repo_id, model_kwargs={ "max_length":int(len(input_text)), "temperature":0.7})

    template = """Question : {question} \n
                Answer : Let me give you detailed answer"""

    prompt = PromptTemplate(template=template, input_variables=['question'])

    chian = LLMChain(prompt=prompt, llm=llm, verbose=True)

    out = chian.run(input_text)
    return out

def text_generator(input_text):
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=hfid, repo_id=repo_id, model_kwargs={ "min_length":400, "max_length":1000,  "temperature":0.9})


    template = """Question : {question} \n
                Answer : Let me give you detailed answer"""

    prompt = PromptTemplate(template=template, input_variables=['question'])

    chian = LLMChain(prompt=prompt, llm=llm, verbose=True)

    out = chian.run(input_text)
    return out



def LLMModel(input_text):
    repo_id = "tiiuae/falcon-7b-instruct"
    llm = HuggingFaceHub(huggingfacehub_api_token=hfid, repo_id=repo_id)

    template = """Question : {question} \n
                   Answer : Let me give you detailed answer"""

    prompt = PromptTemplate(template=template, input_variables=['question'])

    chian = LLMChain(prompt=prompt, llm=llm, verbose=True)

    out = chian.run(input_text)
    return out




# Streamlit app layout
st.title("ASK ME - IVY")

# Sidebar with task selection
selected_task = st.sidebar.selectbox("Select NLP Task:", [ "Code", "Bookish Summary",  "Text Generation"])

# Input text area
input_text = st.text_area("Enter Text:")

# Buttons to trigger tasks
if st.button("Process"):
    if selected_task == "Bookish Summary" and input_text:
        result = summarize_text(input_text)
        st.write(result)

    elif selected_task == "Text Generation" and input_text:
        result = text_generator(input_text)
        st.write(result)

    elif selected_task == "Code" and input_text:
        result = LLMModel(input_text)
        st.write(result)


    else:
        st.info("Please enter text and select a task from the sidebar.")
