import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from io import BytesIO

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title='AI Data Science Tutor', page_icon="📊", layout='wide')

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7, google_api_key=api_key)

# Initializing memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)




# System message
system_message = SystemMessage(
    content=f"You are an AI tutor specialized in answering only Data Science-related questions. "
            f"If the user asks anything outside Data Science, politely refuse to answer. "
            
)

st.title("🔍 Ask AI: Your Data Science Helper")

# Display chat history
chat_history = []
for msg in st.session_state.memory.chat_memory.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(f"**AI:** {msg.content}")

# User input
user_query = st.chat_input("Ask me anything about Data Science...")

if user_query:
    conversation_history = [system_message] + st.session_state.memory.chat_memory.messages + [HumanMessage(content=user_query)]
    ai_response = chat_model.invoke(conversation_history)

    st.session_state.memory.chat_memory.add_user_message(user_query)
    st.session_state.memory.chat_memory.add_ai_message(ai_response.content)

    # Display user message
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_query}")
    
    # Display AI Response
    with st.chat_message("assistant"):
        st.markdown(f"**AI:** {ai_response.content}")

    chat_history.append(f"**You:** {user_query}")
    chat_history.append(f"**AI:** {ai_response.content}")

# Function to save chat history as TXT file
def generate_txt(chat_text):
    txt_buffer = BytesIO()
    txt_buffer.write(chat_text.encode("utf-8"))
    txt_buffer.seek(0)
    return txt_buffer

# Export chat history as TXT file
if chat_history:
    chat_text = "\n".join(chat_history)
    txt_file_path = generate_txt(chat_text)
    st.sidebar.download_button("📥 Download Chat History", txt_file_path, file_name="chat_history.txt", mime="text/plain")

# Reset chat button
if st.button("Reset Chat"):
    st.session_state.memory.clear()
    st.rerun()
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;  /* Add this to stretch from left */
            right: 0; /* Add this to stretch from right */
            width: 100%; 
            text-align: center;
            padding: 10px;
            background-color: #f1f1f1;
            color: #333333;
            font-weight: bold;
            border-top: 2px solid #cccccc;
            z-index: 9999; /* Keeps it on top */
        }
    </style>
    <div class="footer">
        Created by Neelima 🚀
    </div>
    """,
    unsafe_allow_html=True
)
