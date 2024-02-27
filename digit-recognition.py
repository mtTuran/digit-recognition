import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
import joblib
from collections import Counter

'''
# digit recognition with neural network
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, validation_split=0.3)

# Plot training accuracy over epochs
epochs = range(1, len(history.history['accuracy']) + 1)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], label='Testing Accuracy')
plt.title('Model Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test Metrics
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: {0}\nLoss: {1}".format(accuracy, loss))

model.save('digits_3.model')
'''
'''
# digit recognition with conv. neural network
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_trainr = np.array(x_train).reshape(-1, 28, 28, 1) # increasing one dimension for kernel-filter operation
x_testr = np.array(x_test).reshape(-1, 28, 28, 1) # increasing one dimension for kernel-filter operation

cnn_model = tf.keras.models.Sequential()
# first conv layer
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape = x_trainr.shape[1:], activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
# second conv layer
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape = x_trainr.shape[1:], activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
# second conv layer
cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape = x_trainr.shape[1:], activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))
# fully-connected layer
cnn_model.add(tf.keras.layers.Flatten())
cnn_model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.Dense(units=32, activation=tf.nn.relu))
cnn_model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_history = cnn_model.fit(x_trainr, y_train, epochs=5, validation_split = 0.3)
cnn_model.save('cnn_digit_rec.model')

# Plot training accuracy over epochs
epochs = range(1, len(cnn_history.history['accuracy']) + 1)
plt.plot(epochs, cnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(epochs, cnn_history.history['val_accuracy'], label='Testing Accuracy')
plt.title('CNN Model Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Test Metrics
loss, accuracy = cnn_model.evaluate(x_testr, y_test)
print("Accuracy: {0}\nLoss: {1}".format(accuracy, loss))
'''

'''
# digit recognition with kNN
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

kNN_1_model = KNeighborsClassifier(n_neighbors=1)
kNN_3_model = KNeighborsClassifier(n_neighbors=3)
kNN_5_model = KNeighborsClassifier(n_neighbors=5)

kNN_1_model.fit(x_train, y_train)
kNN_3_model.fit(x_train, y_train)
kNN_5_model.fit(x_train, y_train)

joblib.dump(kNN_1_model, 'digits_kNN_1.joblib')
joblib.dump(kNN_3_model, 'digits_kNN_3.joblib')
joblib.dump(kNN_5_model, 'digits_kNN_5.joblib')

# kNN Model Evaluation
scores = []
k_values = [1, 3, 5]
scores.append(kNN_1_model.score(x_test, y_test))
scores.append(kNN_3_model.score(x_test, y_test))
scores.append(kNN_5_model.score(x_test, y_test))
print("kNN_1_model score: {}\nkNN_3_model score: {}\nkNN_5_model score: {}".format(scores[0], scores[1], scores[2]))

# Plot kNN model scores
plt.bar(k_values, scores, color=['red', 'green', 'blue'])
plt.title('kNN Model Scores')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1.01, 0.1))
plt.show()
'''

'''
# NN manuel test
model = tf.keras.models.load_model('digits_3.model')

for i in range(1, 8):
    img = cv.imread(f'manuel-test/{i}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print("The number is probably:", np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
'''

'''
# kNN manuel test
kNN_1_model = joblib.load('digits_kNN_1.joblib')
kNN_3_model = joblib.load('digits_kNN_3.joblib')
kNN_5_model = joblib.load('digits_kNN_5.joblib')
predictions = []
for i in range(1, 8):
    img = cv.imread(f'manuel-test/{i}.png', cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (28, 28))  # Resize to match the training data
    img = np.invert(np.array([img]))
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(-1)

    # Make prediction
    prediction = kNN_5_model.predict([img])
    predictions.append(prediction[0])

    # Display the image and prediction
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted Label: {prediction[0]}")
    plt.show()
'''
    
'''
# cnn manuel test
model = tf.keras.models.load_model('cnn_digit_rec.model')
predictions = []
for i in range(1, 8):
    img = cv.imread(f'manuel-test/{i}.png', cv.IMREAD_GRAYSCALE)        # create some test images in png format and name them "1", "2", "3"....
    img = cv.resize(img, (28, 28))  # Resize to match the model's input shape
    img = np.invert(np.array([img]))
    img = img / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predictions.append(predicted_label)

    # Display the image and prediction
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.title(f"Predicted Label: {predicted_label}")
    plt.show()
'''
