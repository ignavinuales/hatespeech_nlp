from cleaning import clean_data
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, SpatialDropout1D, Dropout, LSTM
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras import callbacks
from keras.utils import pad_sequences
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf


df = pd.read_csv('measuring_hate_speech.csv')

df['clean_text'] = df['text'].apply(clean_data)

# Split into features and target values
features = np.array(df['clean_text'].values)
labels = np.array(df['hatespeech'].values)

# # Encode labels --> 0 = negative, 1 = positive
# encoder = LabelBinarizer()
# labels = encoder.fit_transform(labels)

# Tokenazation
vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(features)
vocab_size = len(tokenizer.word_index) + 1

# Save tokenizer
with open('tokenizer.pickle', 'wb') as file:
    pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

# integer encode the reviews.
encoded_features = tokenizer.texts_to_sequences(features)

# pad reviews to a max length of 150 words
max_length = 50
padded_features = pad_sequences(encoded_features, maxlen=max_length, padding='post')

# Split into train and test dataset, and check size
X_train, X_test, y_train, y_test = train_test_split(padded_features, labels, test_size=0.25, shuffle=True, stratify=labels)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


y_train, y_test = y_train.astype('float32').reshape(-1,1), y_test.astype('float32').reshape(-1, 1)


# Model
model = 0
tf.keras.backend.clear_session()
model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=max_length))
model.add(SpatialDropout1D(0.3))
# While passing output to next LSTM layer set return_sequence to Ture.
model.add(Bidirectional(LSTM(120, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.25))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='relu'))


# model.add(Flatten())
model.add(Dense(3,activation=tf.keras.activations.softmax))
# model.summary()

batch_size = 64
epochs = 10

early_stop = callbacks.EarlyStopping(monitor='val_loss', 
                                     verbose=0,
                                     patience=5,
                                     mode='auto',
                                     restore_best_weights=True)

checkpoint = callbacks.ModelCheckpoint(filepath='checkpoints/model_1_checkpoint.h5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[early_stop, checkpoint])