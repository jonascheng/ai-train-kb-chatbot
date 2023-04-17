from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

from pathlib import Path

import gradio as gr
import openai
import os

max_input_size = 2000
num_output = 500
max_chunk_overlap = 0
chunk_size_limit = 2000
persist_directory = 'chromadb'

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#text-search-embedding
embeddings = OpenAIEmbeddings(
    document_model_name="text-search-davinci-doc-001",
    query_model_name="text-search-davinci-query-001",
    # set chunk_size 1, ref: https://github.com/hwchase17/langchain/issues/1560
    chunk_size=1)


def construct_db(directory_path):
    loader = DirectoryLoader(directory_path, glob='**/*.pdf')
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size_limit,
                                          chunk_overlap=max_chunk_overlap)
    ll_docs = text_splitter.split_documents(documents)

    db = Chroma.from_documents(documents=ll_docs,
                               embedding=embeddings,
                               persist_directory=persist_directory)

    db.persist()

    return db


def chatbot(input_text):
    llm = AzureOpenAI(deployment_name="text-davinci-003",
                      temperature=0,
                      model_kwargs={
                          "api_type": "azure",
                          "api_version": "2022-12-01",
                          "api_base": os.getenv('OPENAI_API_BASE'),
                          "api_key": os.getenv("OPENAI_API_KEY"),
                      })

    db = Chroma(persist_directory=persist_directory,
                embedding_function=embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=retriever)

    response = qa({"query": input_text})

    return response['result']


iface = gr.Interface(fn=chatbot,
                     inputs=gr.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

# if index not exist, then build it
path = Path(persist_directory)
if not path.exists():
    # create chroma index directory
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    index = construct_db("docs")

iface.launch(share=True, server_name="0.0.0.0", server_port=8080)
