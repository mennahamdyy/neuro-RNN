# neuro-RNN
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text = "I love deep learning"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

X = np.array(sequences[:3]).reshape((1, 3))  
y = np.array(sequences[3])  

vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length=3))
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, np.array([y]), epochs=500, verbose=0)

prediction = model.predict(X, verbose=0)
predicted_word_index = np.argmax(prediction)
predicted_word = [word for word, index in tokenizer.word_index.items() if index == predicted_word_index]

print("Expected:", tokenizer.index_word[int(y)])
print("Predicted:", predicted_word[0] if predicted_word else "Unknown")
