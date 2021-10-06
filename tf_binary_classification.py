# https://www.tensorflow.org/tutorials/keras/text_classification
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as tfl
import numpy as np
print(tf.__version__)

# imdb dataset
# imdb는 각 단어가 해당하는 정수로 바뀐(a == 0, zulu = 9999 등등) 리뷰 문장과,
# 긍정/부정의 결과를 포함하는 라벨이 한쌍으로 이루어진 데이터셋입니다.
# 단어가 정수로 바뀐 이유는 학습 효율성을 높이기 위해서입니다.

# imdb.load_data(num_words=k) 에서
# num_words는 리뷰를 상위 k번째 단어까지만 포함하고 그 아래는 oov_char로 표현합니다.
# 예를 들어 잘 쓰이지 않는 durian같은 경우 그대로 표현되지 않고 oov_char로 표현합니다.
# 이는 학습 효율을 높이기 위함입니다.(특이한 단어에 과적합 방지)
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# data shape을 확인합니다.
print("train data shape: {}, train_labels_shape: {}".format(train_data.shape, train_labels.shape))
# >>> train data shape: (25000,), train_labels_shape: (25000,)
for dt in range(5):
    print(len(train_data[dt]), end=' ')
# >>> 218 189 141 550 147
print(max([len(i) for i in train_data]))
# >>> 2494

# 확인 결과 리뷰마다 길이가 다릅니다.
# 따라서 input shape를 일정하게 만들어주는 데이터 전처리가 필요합니다.

# 그 전에 정수로 표현되어있는 리뷰를 단어로 다시 바꿀 수 있는 함수를 만들겠습니다.
# key = word, value = index 인 dictionary. index는 1부터 시작합니다.
word_index = imdb.get_word_index()
# 맨 앞에 직접 설정으로 4개 단어를 추가하고 싶기 때문에 dictionary의 index를 전체적으로 뒤로 미루겠습니다.
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
# key = index, value = word 인 dictionary
reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reversed_word_index.get(i, '?') for i in text])
    # get() i가 있다면 i의 value를, 없다면 '?'를 리턴


print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 위에서 단어를 상위 10000개만 불러왔기 때문에 사전 크기는 10000개입니다.
vocab_size = 10000
model = keras.Sequential([
    # word embedding으로 여러 단어들을 16개의 특징으로 묶는다.
    # 그리고 단어가 해당 특징을 얼마나 가지는지를 실수값으로 표현.
    tfl.Embedding(vocab_size, 16, input_shape=(None,)),
    # return (batch, sequence, embedding)

    # global average pooling 1d는 두번째 차원에 대한 평균을 구해 반환합니다.
    tfl.GlobalAveragePooling1D(),
    # (batch, sequence, embedding) -> (batch, 1, embedding) -> (batch, embedding)

    tfl.Dense(16, activation='relu'),
    # 2진분류(1/0)를 할 예정이므로 sigmoid를 써줍시다.
    tfl.Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

history_dict = history.history
print(history_dict.keys())

# 그래프로 확인해봅시다.
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 그림을 초기화합니다

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
