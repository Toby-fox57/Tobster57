import os
import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Toby code
def image_data(file_list):
    file_len, target = np.empty(0), np.empty(0)  # Creates an empty arrays to store the file lengths and the target data
    data = np.empty((0, 45, 45))  # Creates a 3D empty array to store data

    print(file_list)

    for file_id, digit in enumerate(file_list):

        print(digit)

        file_len = np.append(file_len, len(os.listdir(file_directory + digit)))  # Appends the file length
        target = np.append(target, np.linspace(file_id, file_id, file_len[file_id].astype(
            int)))  # Appends the target data of each character

        imgs = np.ndarray((file_len[file_id].astype(int), 45, 45),
                          dtype='uint8')  # Creates an empty array to store all the image data with predefined size

        for count, img in enumerate(os.listdir(file_directory + digit)):
            imgs[count][:, :] = cv2.imread(file_directory + digit + '/' + img, 0)  # Reads each jpg in greyscale
            # and stores the data in the array

        data = np.vstack((data, imgs))  # Stacks the image data to the data array

        print("Characters", file_id + 1, "/", len(file_list), "processed")

    return file_len, target, data


def train_and_test(target, data):
    x_train, x_test, y_train, y_test = train_test_split(data, target,
                                                        test_size=0.2)  # Split the test data and training data

    #x_train, x_test = tf.keras.utils.normalize(x_train, axis=1), tf.keras.utils.normalize(x_test,
                                                                                          #axis=1)  # Normalise the test and train data

    return x_train, x_test, y_train, y_test

file_directory = 'extracted_images/'
file_list = os.listdir(file_directory)  # Get the file names in the directory

print("number of GPUs available: ", len(tf.config.list_physical_devices('GPU')))

file_len, target, data = image_data(file_list)
x_train, x_test, y_train, y_test = train_and_test(target, data)

# Wardii code
dataset_size = np.sum(file_len)
number_of_classes = file_len.size
image_shape = (45, 45)


def calculate_output_filters(input_volume, kernel_size, stride, padding):
    return ((input_volume - kernel_size + 2 * padding) / stride) + 1


input_size = 45
output_filters = calculate_output_filters(input_size, 3, 1, 0)

conv1 = tf.keras.layers.Conv1D(output_filters, 3, 1, input_shape=image_shape)
output_filters = calculate_output_filters(output_filters, 3, 1, 0)
dense1 = tf.keras.layers.Dense(32, activation='relu')
conv2 = tf.keras.layers.Conv1D(output_filters, 3, 1)
output_filters = calculate_output_filters(output_filters, 3, 1, 0)
dense2 = tf.keras.layers.Dense(32, activation='relu')
maxPool = tf.keras.layers.MaxPool1D(2)
output_filters /= 2
dense3 = tf.keras.layers.Dense(32, activation='relu')
conv3 = tf.keras.layers.Conv1D(output_filters, 3, 1)
output_filters = calculate_output_filters(output_filters, 3, 1, 0)
dense4 = tf.keras.layers.Dense(32, activation='relu')
output_filters /= 2

model = tf.keras.models.Sequential([  # New Model
    conv1,
    dense1,
    conv2,
    dense2,
    maxPool,
    dense3,
    conv3,
    dense4,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(number_of_classes, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=3)  # Accuracy = 97.44%
model.evaluate(x_test, y_test, batch_size=16)

model.save('saved_model/my_model')