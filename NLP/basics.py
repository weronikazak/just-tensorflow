import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers as ly
from tensorflow.keras.models import Sequential

sentences = [
	'I love my dog',
	'I love hot dog',
	'You love my dog!',
	'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<START>")
# oov_token = sign to recognize unseen in data (sentences) words
# wszystkie eteksty musza mieć takie same wymiary, od tego służy padding
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index;

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding="post", truncating="post")
# opcja maxlen zmniejsza tekst paddingowany

print(word_index)
print(sequences)
print(padded)

# --------------------
vocab_size = 1000
embedding_dim = 16
max_length = 16
training_size = 2000



model = Sequential([
	ly.Embedding(vocab_size, embedding_dim, input_length=max_length),
	ly.Flatten(),
	# ly.Flatten() jest podoby do ly.GlobalAveragePooling1D(), można go użyć naprzemiennie
	ly.Dense(6, activation="relu"),
	ly.Dense(1, activation="sigmoid")
])

