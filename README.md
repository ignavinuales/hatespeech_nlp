# Hatespeech Detection using LSTM Neural Networks (API & Web App)
This project aimed at developing an API and Web App that are able to detect hate speech from text. 

Main technologies used:
    - TensorFlow
    - FastAPI
    - Oracle Cloud Infrastructure (Virtual Machines for deploying the REST API on Oracle servers)
    - Oracle SQL (Oracle Cloud Database)
    - NLTK (natural language processing)

## Project Directory Structure

    .
    ├── Models         
        ├── LSTM_checkpoint.h5  # LSTM model used for inference
    ├── cleaning.py             # python module that processes text before running a model inference
    ├── .env                    # used to store and hide credentials to connect to Oracle SQL Database
    ├── inference.py            # python module to run model inference
    ├── api_prediction.py       # python module with REST API for running model inference
    ├── update_db.py            # python module that updates the SQL database 
    ├── requirements.txt        # requirements file with all libraries/modules used for building the project
    ├── tokenizer.pickle        # pickle file that stores the tockenizer object used when training the model
    ├── webapp.py               # python module for running the front-end. Built on Streamlit framework
    └── README.md

## Motivation
The goal of hate speech detection is to determine if communication contains hatred or promotes violence against an individual or a group of individuals. Prejudice against "protected traits" such their ethnicity, gender, sexual orientation, religion, age, etc., is typically the basis for this.

### Steps for the machine learning part
1. **Database¹**: 135,000 tweets were used for training the LSTM model. The tweets were labeled hate speech, supportive speech, and neutral or ambiguous (three classes).
2. **Data processing**: the text was processed and tokenized so that it is clean and the machine learning model can handle it.
3. **LSTM model training**: the deep learning model was trained using TensorFlow. Then, model fine-tuning to improve validation accuracy. 

### Steps for MLOps
1. **Building REST API for model inference**. The API receives a text, runs the model (called inference), and returns the prediction. Here we find three modules:
    a. cleaning.py: used for processing the input data. 
    b. inference.py: tokening the processed data, running the model, and getting the output.
    c. predict_api.py: where the main body (with POST request) from the FastAPI framework resides.

2. **Deploying REST API on Oracle Cloud**. The API was deployed on a personal Oracle instance (virtual machine on the cloud).
3. **Building frontend**. I created a Web App for users to test the model and for me to monitor its performance in real life. As the Web App is a Minimum Viable Product (MVP), I built it using Streamlit and hosted it on Streamlit Cloud.
4. **Model monitoring**. Text inputs from users using the Web App are stored in a SQL database on the Oracle Cloud. Users can also give feedback on the prediction, which is also stored in the database. Thus, I can monitor the model's performance and find weak points to retrain the LSTM model later.