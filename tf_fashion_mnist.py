import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

# Data loading
# tensorflow서버로부터 fashion_mnist를 불러옵니다.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

'''
print("image dataset shape =", train_images.shape)
'''
# Set labels name
# 데이터넷에는 라벨 이름이 들어가있지 않으므로 이름을 저장하는 리스트를 만들겠습니다.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data preprocessing
# 데이터의 input 값들이 0~255 사이의 값들입니다.
# 연산을 수월하게 하기 위해 255로 나누어 0~1 사이로 만듭니다.
train_images, test_images = train_images/255.0, test_images/255.0

# Dataset Check
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

# Model develop
model = keras.Sequential([
    tfl.Flatten(input_shape=(28, 28)),
    tfl.Dense(units=128, activation='relu'),
    tfl.Dense(units=10, activation='softmax')
])

# Model compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model training
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\n테스트 정확도:', test_acc)

# Prediction
# train된(fit된) model에 test data를 넣어 prediction을 한다.
predictions = model.predict(test_images)
# pred_of_i = np.argmax(predictions[i])
pred_acc = [1 if np.argmax(predictions[i]) == test_labels[i] else 0 for i in range(len(predictions))]
print("prediction accuracy: ",sum(pred_acc)/len(pred_acc)*100)
print(predict0,test_labels[0])


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()