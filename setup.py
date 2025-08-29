from setuptools import setup, find_packages

setup(
    name="research-assistant",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.267",
        "langchain-openai>=0.0.2",
        "openai>=1.3.0",
        "pymupdf>=1.23.0",
        "pdfplumber>=0.10.2",
        "pinecone-client>=2.2.4",
        "sentence-transformers>=2.2.2",
        "streamlit>=1.28.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.0",
        "llama-index>=0.8.0",
        "pypdf>=3.15.1",
        "pymupdf>=1.23.0"
    ],
    test_suite="tests",
    tests_require=["pytest"],
)