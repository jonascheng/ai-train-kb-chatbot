FROM python:3.9-slim-bullseye

RUN python -m pip install -U pip
RUN pip install openai gpt_index PyPDF2 gradio

WORKDIR /app
COPY . .

CMD [ "python", "/app/app.py" ]