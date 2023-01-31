FROM python:3.10.6-slim-buster
WORKDIR /app
COPY api_prediction.py cleaning.py inference.py requirements.txt tokenizer.pickle /app/
COPY ./Models /app/Models
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["uvicorn", "api_prediction:app", "--host", "0.0.0.0", "--port", "8000"]