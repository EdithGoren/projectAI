"""
Research Assistant - Streamlit UI

A web interface for the Research Assistant multi-agent system.
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from research_assistant.vector_database import VectorStore, EmbeddingGenerator
from research_assistant.agents import Orchestrator

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("🧠 AI Research Assistant")
st.markdown("""
    An intelligent research assistant that helps you process documents and answer questions.
    Built with LangChain and powered by a multi-agent system.
""")

# Initialize session state
if "orchestrator" not in st.session_state:
    # Check for API keys
    if "OPENAI_API_KEY" not in os.environ:
        st.error("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    if "PINECONE_API_KEY" not in os.environ:
        st.error("PINECONE_API_KEY not found in environment variables. Please check your .env file.")
    
    # Initialize vector store and orchestrator
    try:
        vector_store = VectorStore(
            api_key=os.environ.get("PINECONE_API_KEY"),
            index_name="research",
            namespace="default",
            embedding_model="openai"
        )
        st.session_state.orchestrator = Orchestrator(vector_store=vector_store)
        st.session_state.messages = []  # Initialize message history
        st.session_state.document_count = 0  # Track number of processed documents
    except Exception as e:
        st.error(f"Error initializing the application: {str(e)}")

# Sidebar for document processing
with st.sidebar:
    st.header("Document Processing")
    st.markdown("Upload documents to the research assistant.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to upload"
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                # Process the document
                try:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = st.session_state.orchestrator.process_documents(temp_file_path)
                        st.session_state.document_count += 1
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"✅ Processed document: {uploaded_file.name}"
                        })
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # Update progress
                progress.progress((i + 1) / len(uploaded_files))
                
                # Remove temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            st.success(f"Successfully processed {len(uploaded_files)} document(s)!")

    # Document stats
    st.subheader("Document Statistics")
    st.markdown(f"**Documents processed:** {st.session_state.document_count}")
    
    st.divider()
    st.markdown("Created by [Edith Goren]")

# Main chat interface
st.header("Ask Your Research Questions")
st.markdown("Ask questions about the documents you've uploaded or any research topic.")

# Input for research questions
user_question = st.text_input(
    "Your question:",
    placeholder="What is machine learning?",
    help="Ask any research question here"
)

# Process the question when submitted
if user_question:
    # Add user question to message history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Display "Researching..." message
    with st.spinner("🔍 Researching your question..."):
        try:
            # Call orchestrator to research the question
            result = st.session_state.orchestrator.research_question(user_question)
            
            # Add response to message history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["response"]
            })
            
            # Add bibliography if available
            if "bibliography" in result and result["bibliography"]:
                bibliography = "\n\n**References:**\n" + "\n".join(result["bibliography"])
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bibliography,
                    "is_bibliography": True
                })
        except Exception as e:
            st.error(f"Error researching question: {str(e)}")
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"❌ I encountered an error while researching: {str(e)}"
            })

# Display message history
st.subheader("Conversation")
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:  # assistant message
        with st.chat_message("assistant"):
            # Check if this is a bibliography message
            if message.get("is_bibliography", False):
                with st.expander("Show References"):
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])

# Add a clear conversation button
if st.session_state.messages and st.button("Clear Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

# Run the app: streamlit run app_ui.py
