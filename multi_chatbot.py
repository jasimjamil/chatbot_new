from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from typing import Optional
import requests
import os
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import streamlit as st

# Environment Configuration
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "default_groq_key")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY", "default_google_key")

# FastAPI Initialization
app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
db = {}  # In-memory database for demo purposes

# Pydantic Models
class User(BaseModel):
    username: str
    password: str

class UserInDB(User):
    hashed_password: str

# Utility Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# FastAPI Routes
@app.post("/signup", response_model=User)
async def signup(user: User):
    if user.username in db:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    hashed_password = hash_password(user.password)
    db[user.username] = UserInDB(username=user.username, hashed_password=hashed_password, password=user.password)
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = db.get(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return {"access_token": form_data.username, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = db.get(token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return {"username": user.username}

# Streamlit Chatbot Dashboard
st.set_page_config(page_title="Chatbot Dashboard", page_icon="🤖", layout="wide")

if "chatbots" not in st.session_state:
    st.session_state["chatbots"] = {}
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def create_chatbot(name, description, pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Split text into chunks and generate embeddings
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(docs, embeddings)
    vectorstore.save_local(f"chatbot_{name}_vectorstore")

    # Define chatbot prompt template
    prompt_template = f"""
    You are {name}, a chatbot designed to assist users. Your expertise is {description}.
    Always provide clear and friendly answers. Use the following context to answer questions:
    Document Context: {{context}}
    Question: {{question}}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create conversational retrieval chain
    chat_model = ChatGroq(model="mixtral-8x7b-32768")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )

    st.session_state["chatbots"][name] = {
        "description": description,
        "qa_chain": qa_chain
    }

# Authentication Flow
def signup(username, password):
    response = requests.post("http://127.0.0.1:8000/signup", json={"username": username, "password": password})
    if response.status_code == 200:
        st.success("Signup successful!")
    else:
        st.error("Signup failed.")

def login(username, password):
    response = requests.post("http://127.0.0.1:8000/token", data={"username": username, "password": password})
    if response.status_code == 200:
        st.session_state["authenticated"] = True
        st.session_state["token"] = response.json()["access_token"]
        st.experimental_rerun()
    else:
        st.error("Invalid credentials.")

if not st.session_state["authenticated"]:
    st.title("Chatbot Dashboard")
    auth_option = st.radio("Authentication:", ["Sign Up", "Log In"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if auth_option == "Sign Up":
        if st.button("Sign Up"):
            signup(username, password)
    else:
        if st.button("Log In"):
            login(username, password)
else:
    st.sidebar.title("Menu")
    option = st.sidebar.radio("Navigate to:", ["Dashboard", "Create Bot", "Chat with Bot"])
    
    if option == "Dashboard":
        st.title("Your Chatbots")
        if st.session_state["chatbots"]:
            for name, bot in st.session_state["chatbots"].items():
                st.subheader(name)
                st.write(f"Description: {bot['description']}")
        else:
            st.write("No chatbots available.")

    elif option == "Create Bot":
        st.title("Create Chatbot")
        name = st.text_input("Chatbot Name")
        description = st.text_area("Description")
        pdf_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if st.button("Create"):
            if name and description and pdf_files:
                create_chatbot(name, description, pdf_files)
                st.success(f"Chatbot '{name}' created.")
            else:
                st.error("Fill in all fields.")

    elif option == "Chat with Bot":
        st.title("Chat with Bot")
        if st.session_state["chatbots"]:
            selected_bot = st.selectbox("Choose a chatbot:", list(st.session_state["chatbots"].keys()))
            query = st.text_input("Ask a question:")
            if st.button("Ask"):
                qa_chain = st.session_state["chatbots"][selected_bot]["qa_chain"]
                result = qa_chain({"question": query})
                st.write(f"{selected_bot}: {result['answer']}")
        else:
            st.write("No chatbots available.")
