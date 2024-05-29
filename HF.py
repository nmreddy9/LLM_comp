import streamlit as st
from transformers import pipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain


def summarize_text(input_text):

    repo_id = "facebook/bart-large-cnn"
    hfid = "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"
    llm = HuggingFaceHub(huggingfacehub_api_token=hfid, repo_id=repo_id, model_kwargs={ "min_length":int(len(input_text)/1.5)})

    template = """Question : {question} \n
                Answer : Let me give you detailed answer"""

    prompt = PromptTemplate(template=template, input_variables=['question'])

    chian = LLMChain(prompt=prompt, llm=llm, verbose=True)

    out = chian.run(input_text)
    return out

def classify_text(input_text):
    classifier = pipeline("sentiment-analysis")
    classification = classifier(input_text)[0]['label']
    return classification

def rephrase_text(input_text):

    repo_id = "humarin/chatgpt_paraphraser_on_T5_base"
    hfid = "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"
    llm = HuggingFaceHub(huggingfacehub_api_token=hfid, repo_id=repo_id, model_kwargs={"min_length":int(len(input_text) - len(input_text)*0.3), "max_length":int(len(input_text)),  "temperature":0.7})


    template = """Question : {question} \n
                Answer : Let me give you detailed answer"""

    prompt = PromptTemplate(template=template, input_variables=['question'])

    chian = LLMChain(prompt=prompt, llm=llm, verbose=True)

    out = chian.run(input_text)
    return out


def LLMModel(input_text):
    repo_id = "tiiuae/falcon-7b-instruct"
    hfid = "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"
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
selected_task = st.sidebar.selectbox("Select NLP Task:", [ "Code", "Summarization", "Text Classification", "Text Rephrasing"])

# Input text area
input_text = st.text_area("Enter Text:")

# Buttons to trigger tasks
if st.button("Process"):
    if selected_task == "Summarization" and input_text:
        st.subheader("Summary:")
        result = summarize_text(input_text)
        st.write(result)


    elif selected_task == "Text Classification" and input_text:
        st.subheader("Classification:")
        result = classify_text(input_text)
        st.write(f"The text is classified as: {result}")

    elif selected_task == "Text Rephrasing" and input_text:
        st.subheader("Text Rephrasing:")
        result = rephrase_text(input_text)
        st.write(result)

    elif selected_task == "Code" and input_text:
        st.subheader("Code:")
        result = LLMModel(input_text)
        st.write(result)

    else:
        st.info("Please enter text and select a task from the sidebar.")