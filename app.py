"""
Interactive Research Assistant Application

This script provides a simple command-line interface to interact with the research assistant.
"""

import os
import sys
from dotenv import load_dotenv
from research_assistant.vector_database import VectorStore, EmbeddingGenerator
from research_assistant.agents import Orchestrator

def main():
    """Run the interactive research assistant."""
    print("\n\n")
    print("="*80)
    print("            RESEARCH ASSISTANT - INTERACTIVE MODE")
    print("="*80)
    print("\nIMPORTANT: This application runs in interactive mode.")
    print("After it starts up, you can type commands and questions directly into the application.")
    print("Please wait patiently for the application to initialize...")
    print("\n")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your OpenAI API key.")
        print("Example: OPENAI_API_KEY=sk-...")
        sys.exit(1)
        
    # Check for Pinecone API key
    if "PINECONE_API_KEY" not in os.environ:
        print("Error: PINECONE_API_KEY environment variable is not set.")
        print("Please create a .env file with your Pinecone API key.")
        print("Example: PINECONE_API_KEY=...")
        sys.exit(1)
    
    # Initialize components
    print("Initializing Research Assistant...")
    
    # Create vector store - this will create the EmbeddingGenerator internally
    vector_store = VectorStore(
        api_key=os.environ.get("PINECONE_API_KEY"),
        index_name="research",  # Make sure this matches your Pinecone index name
        namespace="default",
        embedding_model="openai"  # Using embedding_model parameter instead of embedding_generator
    )
    
    # Create orchestrator
    orchestrator = Orchestrator(vector_store=vector_store)
    
    # Main interaction loop
    print("\n" + "="*70)
    print("🤖 Research Assistant is ready!")
    print("="*70)
    print("- To process documents, type: process [file_or_directory_path]")
    print("  Example: process data/raw/example.pdf")
    print("  Another example: process data/raw/")
    print()
    print("- To ask research questions, simply type your question")
    print("  Example: What are the basic principles of machine learning?")
    print("  Another example: What is contrastive learning?")
    print()
    print("- To exit, type 'exit' or press Ctrl+C")
    print("="*70 + "\n")
    
    try:
        while True:
            # Get user input
            user_input = input("\n🧠 > ")
            
            # Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using Research Assistant. Goodbye!")
                break
            
            # Process the user input
            try:
                # Check if this is a document processing request
                if user_input.lower().startswith("process "):
                    # Extract the path from the command
                    path = user_input[8:].strip()
                    print(f"\nProcessing documents at: {path}")
                    
                    # Process documents
                    result = orchestrator.process_documents(path)
                    
                    # Display result
                    print("\n✅ Documents processed successfully!")
                    
                # Otherwise, treat as a research question
                else:
                    print("\nResearching your question...")
                    
                    # Send the question to the orchestrator
                    result = orchestrator.research_question(user_input)
                    
                    # Display the response
                    print("\n📝 Answer:")
                    print(result["response"])
                    
                    # Display sources if available
                    if "bibliography" in result and result["bibliography"]:
                        print("\n📚 Sources:")
                        for i, source in enumerate(result["bibliography"]):
                            print(f"  [{i+1}] {source}")
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nThank you for using Research Assistant. Goodbye!")

if __name__ == "__main__":
    main()
