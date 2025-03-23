import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
load_dotenv()
from google.generativeai import configure

configure(api_key=os.getenv("GEMINI_API_KEY"))
def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context and the previous conversation history. 
     make sure to provide all the details .If the answer is not in the provided context or chat history, provide responce according to previous conversation history,  but don't provide the wrong answer.also response should look like beautiful and in in well structured manner .
    
    Chat History:\n {chat_history}\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def chatbot():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.title("📄 Chat with Website")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_question = st.chat_input("Ask a question from the PDF files...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            new_db = FAISS.load_local("brainlox_vector_db", embeddings ,allow_dangerous_deserialization = True )
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            
            # Prepare chat history
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            
            # Updated method call to include chat history
            response = chain.invoke({"input_documents": docs, "chat_history": chat_history, "question": user_question})
            reply = response['output_text']
            
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    chatbot()