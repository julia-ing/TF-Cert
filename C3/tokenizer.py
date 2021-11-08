import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',  # 소문자로 모두 바꿔줌
    'You love my dog!',  # 구두점은 영향을 미치지 않음
    'Do you think my dog is amazing?'
]

# tokenizer = Tokenizer(num_words=100)
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)  # tokenizer 초기화
word_index = tokenizer.word_index
# print(word_index)  # {'love': 1, 'my': 2, 'i': 3, 'dog': 4, 'cat': 5, 'you': 6}

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)
# padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)

print(word_index)
print(sequences)
print(padded)

# {'<OOV>': 1, 'my': 2, 'love': 3, 'dog': 4, 'i': 5, 'you': 6, 'cat': 7, 'do': 8, 'think': 9, 'is': 10, 'amazing': 11}
# [[5, 3, 2, 4], [5, 3, 2, 7], [6, 3, 2, 4], [8, 6, 9, 2, 4, 10, 11]]
"""
[[ 0  0  0  5  3  2  4]
 [ 0  0  0  5  3  2  7]
 [ 0  0  0  6  3  2  4]
 [ 8  6  9  2  4 10 11]]
"""

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)  # [[5, 1, 3, 2, 4], [2, 4, 1, 2, 1]]
