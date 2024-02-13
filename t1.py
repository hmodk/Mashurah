import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
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

# Set up vector database
vectordb = DocArrayInMemorySearch.from_documents(all_docs, embeddings)

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
    <div class="message">{}</div>
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
        color: white;
        background-color:#dacec2; /* Green background color */
        border: 2px solid #dacec2; /* Green border */
        border-radius: 10px; /* Rounded corners */
        padding: 10px; /* Add some padding */
    }
    </style>
    """

    # Apply the CSS styles
    st.markdown(TEXT_INPUT_STYLE, unsafe_allow_html=True)

    # Display the text input widget
    user_input = st.text_input("أدخل رسالتك هنا...")

    if st.button("إرسال"):
        with st.spinner("جاري التحليل..."):
            response = qa_chain.run(user_input)
            st.session_state['conversation'].append(("User", user_input))
            st.session_state['conversation'].append(("ChatGPT", response))
            display_conversation()
            # Clear the text input after the button is pressed
            # user_input = ""  # Set the text input to an empty string

# Display conversation history
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
