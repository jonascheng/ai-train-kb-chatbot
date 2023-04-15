from gpt_index import (SimpleDirectoryReader, GPTSimpleVectorIndex,
                       LLMPredictor, PromptHelper, ServiceContext,
                       LangchainEmbedding)
from langchain.chat_models import ChatOpenAI
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings
from pathlib import Path

import gradio as gr
import openai
import os

max_input_size = 2046
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 500


def construct_index(directory_path):
    llm = AzureOpenAI(deployment_name="text-davinci-003",
                      model_kwargs={
                          "api_type": "azure",
                          "api_version": "2022-12-01",
                          "api_base": os.getenv('OPENAI_API_BASE'),
                          "api_key": os.getenv("OPENAI_API_KEY"),
                      })

    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#text-search-embedding
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            document_model_name="text-search-davinci-doc-001",
            query_model_name="text-search-davinci-query-001",
        ))

    prompt_helper = PromptHelper(max_input_size,
                                 num_outputs,
                                 max_chunk_overlap,
                                 chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                   embed_model=embedding_llm,
                                                   prompt_helper=prompt_helper)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index


def chatbot(input_text):
    llm = AzureOpenAI(deployment_name="text-davinci-003",
                      model_kwargs={
                          "api_type": "azure",
                          "api_version": "2022-12-01",
                          "api_base": os.getenv('OPENAI_API_BASE'),
                          "api_key": os.getenv("OPENAI_API_KEY"),
                      })

    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#text-search-embedding
    embedding_llm = LangchainEmbedding(
        OpenAIEmbeddings(
            document_model_name="text-search-davinci-doc-001",
            query_model_name="text-search-davinci-query-001",
        ))

    prompt_helper = PromptHelper(max_input_size,
                                 num_outputs,
                                 max_chunk_overlap,
                                 chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                                   embed_model=embedding_llm,
                                                   prompt_helper=prompt_helper)

    index = GPTSimpleVectorIndex.load_from_disk(
        'index.json', service_context=service_context)
    response = index.query(input_text, response_mode="compact")
    return response.response


# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

iface = gr.Interface(fn=chatbot,
                     inputs=gr.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

# if index not exist, then build it
path = Path('index.json')
if not path.is_file():
    index = construct_index("docs")

iface.launch(share=True, server_name="0.0.0.0", server_port=8080)
