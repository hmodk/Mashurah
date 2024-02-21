import os
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Set OpenAI API Key
os.environ['OPENAI_API_KEY'] = "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"

# Set Streamlit page configuration
st.set_page_config(
    page_title="مشوره",
    page_icon="https://i.ibb.co/BrrTgCz/logo2.png"
)

# Your image URL
icon_image_url = "https://i.ibb.co/BrrTgCz/logo2.png"

# Display the image in the middle of the page
st.markdown(
    f'<div style="display: flex; justify-content: center;"><img src="{icon_image_url}" width="200"></div>',
    unsafe_allow_html=True
)
import streamlit as st

CSS_STYLES = f"""
<style>
    body {{
        direction: rtl;
    }}
    .stApp {{
        background-image: url("");
        background-size: contain; /* Adjusts the background image to cover the entire container */
        background-position: center; /* Centers the background image */
        background-repeat: no-repeat; /* Prevents the background image from repeating */
    }}
    .chat-message {{
        padding: 17px;
        margin: 10px;
        border-radius: 10px;
    }}

    .chat-message.user {{
        background-color: #dfd7cf;
        align-self: flex-start;
        color: black;
        text-align: right;
    }}

    .chat-message.bot {{
        background-color: #a9927d;
        align-self: flex-end;
        color: white;
        text-align: right;
    }}

    .avatar img {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
    }}
</style>
"""

# Inject the CSS styles into the Streamlit app
st.markdown(CSS_STYLES, unsafe_allow_html=True)


# Define bot and user message templates
BOT_TEMPLATE = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/YtJ5cCB/Screenshot-1445-08-02-at-2-28-00-PM.png" alt="Bot Avatar">
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

def display_msg(msg, role):
    # Store the message in the session state
    st.session_state["messages"].append({'content': msg, 'role': role})

    # Display message based on the role (user or bot)
    if role == 'المستخدم':
        st.write(USER_TEMPLATE.format(msg), unsafe_allow_html=True)
    else:
        st.write(BOT_TEMPLATE.format(msg), unsafe_allow_html=True)

def enable_chat_history(func):
    def wrapper(*args, **kwargs):
        # Clear chat history after switching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass
        if "messages" not in st.session_state:
            st.session_state["messages"] = []   
        
        # Show chat history on UI
        for msg in st.session_state["messages"]:
            if msg['role'] == 'المستخدم':
                st.write(USER_TEMPLATE.format(msg['content']), unsafe_allow_html=True)
            else:
                st.write(BOT_TEMPLATE.format(msg['content']), unsafe_allow_html=True)

        # Call the decorated function
        return func(*args, **kwargs)
    return wrapper


@st.cache_resource
def load_vectordb():
    df = pd.read_csv("data.csv")
    loader = DataFrameLoader(df, page_content_column="page_content")
    all_docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    saveDirectory = 'Store'  # directory where to store the db
    
    # Check if cached results file exists
    cached_results_file = './Store/index.pkl'
    if os.path.exists(cached_results_file):
        vectordb = FAISS.load_local(saveDirectory, embeddings)
    else:
        # Run the expensive function
        vectordb = FAISS.from_documents(all_docs, embeddings)
        # Save results to file
        vectordb.save_local(saveDirectory)
    return vectordb


custom_template = """ 
messages=[
    {
      "role": "system",
      "content": "You are a Socratic tutor. Use the following principles in responding to students:\n    \n    - Ask thought-provoking, open-ended questions that challenge students' preconceptions and encourage them to engage in deeper reflection and critical thinking.\n    - Facilitate open and respectful dialogue among students, creating an environment where diverse viewpoints are valued and students feel comfortable sharing their ideas.\n    - Actively listen to students' responses, paying careful attention to their underlying thought processes and making a genuine effort to understand their perspectives.\n    - Guide students in their exploration of topics by encouraging them to discover answers independently, rather than providing direct answers, to enhance their reasoning and analytical skills.\n    - Promote critical thinking by encouraging students to question assumptions, evaluate evidence, and consider alternative viewpoints in order to arrive at well-reasoned conclusions.\n    - Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a growth mindset and exemplifying the value of lifelong learning."
    },
    {
      "role": "user",
      "content": "Help me to understand the future of artificial intelligence."
    }

"""

class CustomDataChatbot:
    
    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
        self.qa_chain = self.setup_qa_chain()  # Store qa_chain as an attribute
        self.conversation_history = []  # Initialize conversation history list
    
    def setup_qa_chain(self):
        # Usage
        vectordb = load_vectordb()

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
            

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0.1, streaming=True)
        
        qa_chain = ConversationalRetrievalChain.from_llm(llm,
                                                         retriever=retriever,
                                                         memory=memory,
                                                         condense_question_prompt=PromptTemplate.from_template(custom_template),
                                                         verbose=True)
        return qa_chain

    @enable_chat_history      
    def main(self):
        user_query = st.chat_input(placeholder="إسئلني !")
        
        if user_query:
            display_msg(user_query, "المستخدم")
            # Process user query and get response
            response = self.qa_chain.run(user_query)
            # Append user query and assistant's response to conversation history
            self.conversation_history.append(("المساعد", response))
            
            # Display conversation history
            for role, msg in self.conversation_history:
                display_msg(msg, role)

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
