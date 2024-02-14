import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
# from langchain_community.vectorstores import DocArrayInMemorySearch
from pydantic import ValidationError
load_dotenv()

# Get the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if openai_api_key is None:
    raise ValueError("OpenAI API key not found in the environment variables.")


# Load the data
df = pd.read_csv("data.csv")

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
all_docs = DataFrameLoader(df, page_content_column="page_content").load()

saveDirectory = 'Store'  # directory where to store the db

def getVectordb():
    # Check if cached results file exists
    cached_results_file = './Store/index.pkl'
    if os.path.exists(cached_results_file):
        vectordb = FAISS.load_local(saveDirectory, embeddings)
        print("db already saved")
    else:
        # Run the expensive function
        vectordb = FAISS.from_documents(all_docs, embeddings)
        # Save results to file
        vectordb.save_local(saveDirectory)
        print("db just saved")
    return vectordb

# Set up vector database
# vectordb = DocArrayInMemorySearch.from_documents(all_docs, embeddings)

vectordb = getVectordb()

# Define retriever
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up memory for contextual conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Set up LLM and QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)

# Define CSS styles
CSS_STYLES = """
<style>
  body {
        direction: rtl;
        background-color: #dacec2; /* light green background color */
    }

    .chat-message {
        padding: 10px;
        margin: 10px;
        border-radius: 10px;
        text-align: right;
    }

    .chat-message.user {
        background-color: #dfd7cf;
        align-self: flex-start;
        color: black;
    }

    .chat-message.bot {
        background-color: #a9927d;
        align-self: flex-end;
        color: white;
    }

    .avatar img {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }

    .title-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }

    .title-text {
        color: black;
        padding: 10px;
        text-align: center;
    }
</style>
"""

# Define bot and user message templates
BOT_TEMPLATE = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/BrrTgCz/logo2.png" alt="Bot Avatar">
    </div>
    <div class="message" style="color: black;">{}</div>
</div>
"""

USER_TEMPLATE = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/WxQx1t7/placeholder-avatar.jpg" alt="User Avatar">
    </div>
    <div class="message">{}</div>
</div>
"""

# Main function
def main():
    st.set_page_config(page_title="مشوره", page_icon="📄", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

    # Display the icon image and the title
    icon_image_url = "https://i.ibb.co/BrrTgCz/logo2.png"  
    title_text = "مشوره"

    st.image(icon_image_url, width=100)  # Adjust the width as per your preference
    st.markdown(f"<div class='title-container'><h1 class='title-text'>{title_text}</h1></div>", unsafe_allow_html=True)

    # Define CSS styles for the text input
    TEXT_INPUT_STYLE = """
    <style>
    /* Style the text input */
    .stTextInput>div>div>input {
        color: black;
        background-color:#dacec2; /* Green background color */
        border: 2px solid #dacec2; /* Green border */
        border-radius: 10px; /* Rounded corners */
        padding: 10px; /* Add some padding */
    }
    </style>
    """

    # Apply the CSS styles
    st.markdown(TEXT_INPUT_STYLE, unsafe_allow_html=True)


    if 'text' not in st.session_state:
        st.session_state.text = ""

    def update():
        st.session_state.text = st.session_state.text_value

    with st.form(key='my_form', clear_on_submit=True):
        st.text_input(" ", placeholder='أدخل سؤالك هنا', value="", key='text_value')
        submit = st.form_submit_button(label='إرسال', on_click=update)

    if submit:
        with st.spinner("جاري التحليل..."):
            response = qa_chain.run(st.session_state.text)
            st.session_state['conversation'].insert(0, ("ChatGPT", response))
            st.session_state['conversation'].insert(0, ("User", st.session_state.text))
            display_conversation()


def display_conversation():
    for speaker, message in st.session_state['conversation']:
        if speaker == "User":
            st.write(USER_TEMPLATE.format(message), unsafe_allow_html=True)
        else:
            st.write(BOT_TEMPLATE.format(message), unsafe_allow_html=True)

# Initialize conversation history in session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

if __name__ == "__main__":
    main()
