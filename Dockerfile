FROM python:3.9-alpine3.16

RUN python -m pip install -U pip
RUN pip install openai gpt_index PyPDF2 gradio


