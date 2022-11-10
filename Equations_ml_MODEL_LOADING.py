import os
import cv2
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

IMAGE_ADDRESS = '2_div.jpg'
MODEL_ADDRESS = 'saved_model/my_model'
FILE_ADDRESS = 'extracted_images'

IMAGE_SHAPE = (45, 45)


def read_file(file_address=FILE_ADDRESS):
    file_list = os.listdir(file_address)  # Get the file names in the directory

    for id, character in enumerate(file_list):

        if character == 'div':
            file_list[id] = '/'

        elif character == 'times':
            file_list[id] = '*'

    return file_list


def image_crop_and_resize(images, input_image, boundaries, image_shape=IMAGE_SHAPE):
    x, y, w, h = boundaries  # Stores the boundary data

    cropped_image = input_image[y:y + h, x:x + w]  # Crops the orignal image to store the data of the object
    resized_image = cv2.resize(cropped_image, image_shape, interpolation=cv2.INTER_AREA)  # Resizes the image to
    # 45 x 45 so it is the same size as the test model.
    images.append(resized_image)  # Appends the resized images data.

    return images


def detect_objects(image_address=IMAGE_ADDRESS, image_shape=IMAGE_SHAPE):
    input_image = cv2.imread(image_address, 0)  # Reads jpg and stores image data in greyscale

    image_edges = cv2.Canny(input_image, threshold1=200, threshold2=240, apertureSize=5)  # Need to research thresholds

    cv2.imshow('edge', image_edges)
    cv2.waitKey(0)

    (objects, _) = cv2.findContours(image_edges.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)  # Need to research findContours

    object_coordinates = np.empty((0, 4))  # Empty array to store each objects coordinates and size
    images = []  # Stores each objects data

    for num_of_obj, obj in enumerate(objects):

        x, y, w, h = cv2.boundingRect(obj)  # Draws an approximate rectangle around the contoured image and stores it

        # coordinates and size

        if w > 10 and h > 10:  # Images any rectangles smaller then 10 pixels

            boundaries = ([x, y, w, h])  # Stores the boundary data
            object_coordinates = np.vstack((object_coordinates, boundaries))  # Sacks the boundary data

            images = image_crop_and_resize(images, input_image, boundaries)

        elif w > 10 and h <= 10:

            boundaries = ([x, y - 10, w, h + 20])  # Stores the boundary data
            object_coordinates = np.vstack((object_coordinates, boundaries))  # Sacks the boundary data

            images = image_crop_and_resize(images, input_image, boundaries)

        elif w <= 10 and h > 10:

            boundaries = ([x - 10, y, w + 20, h])  # Stores the boundary data
            object_coordinates = np.vstack((object_coordinates, boundaries))  # Sacks the boundary data

            images = image_crop_and_resize(images, input_image, boundaries)

    images = [images[i] for i in
              np.argsort(object_coordinates[:, 0])]  # Orders the images of the objects by x-coordinates

    return images


def predict(images, file_list):
    model = tf.keras.models.load_model(MODEL_ADDRESS)
    model.summary()

    equation = ''

    for img in images:
        image_shape_3d = np.append(1, IMAGE_SHAPE)
        img = np.reshape(img, image_shape_3d)

        predict = model.predict(img)
        id = np.argmax(predict[0])

        equation = (equation + file_list[id])

    return equation


def main():
    file_list = read_file()
    images = detect_objects()

    equation = predict(images, file_list)
    result = eval(equation)

    print(equation, '=', result)

    return 0


main()










