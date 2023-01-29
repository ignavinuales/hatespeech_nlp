from cleaning import clean_data
from keras.models import load_model
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle 
import numpy as np

def inference(text: str) -> str:
    """
    It runs a model inference. But first, it cleans the data using the clean_data function from cleaning.py

    Args:
        text [str]: text we want to use for prediction.
    
    Returns:
        [str]: prediction after running the model (one of the three possible outputs: No hate - Supportive speech, 
               Neutral or ambiguos, Hate speech). 
    """

    outputs_dict = {
        0: 'No hate - Supportive speech',
        1: 'Neutral or ambiguos',
        2: 'Hate speech'
    }

    # Process text data
    text = [clean_data(text, inference_mode=True)]

    # Load tokenizer and apply pad sequences
    with open('tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)

    text_sequences = tokenizer.texts_to_sequences(text)
    padded_text = pad_sequences(text_sequences, maxlen=50, padding='post')

    # Load model
    model = load_model('Models/LSTM_checkpoint.h5')
    
    # Make prediction
    prediction = model.predict(padded_text)
    prediction = np.argmax(prediction)

    return outputs_dict[prediction]
