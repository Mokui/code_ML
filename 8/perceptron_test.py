import timeit
import functools
from itertools import combinations
import operator
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
from functools import reduce
import cv2
import numpy as np

# Define an array of 0 or 1 for each 64 pixels of each images
def create_weight_images():
    weight_images = [[random.uniform(0, 1) for elem in range(0, 64)] for elem in range(0, 10)]
    return weight_images

def calculate_new_weight(image, number, weight_images):
    new_weight_images = weight_images

    # Change weigth of the image between O,1
    changed_image = [(elem/16) for elem in image]

    # Change the weight of the rigth weight image
    new_weight_images[number] = [elem + changed_image[idx] for idx, elem in enumerate(new_weight_images[number])]

    # Get an array of the sum of the sum between weight_images and the image to predict
    predicted_value = predict_right_image(changed_image, weight_images)

    # If the prediction is correct, return the weight_images without changement
    if predicted_value == number:
        return weight_images
    # Else return new_weight_images with the modification of the weigth
    else:
        new_weight_images[predicted_value] = [elem - changed_image[idx] for idx, elem in enumerate(new_weight_images[predicted_value])]
        return new_weight_images

def predict_right_image(image, weight_images):
    weight_images_sum = []
    for idx, elem in enumerate(weight_images):
        weight_images_sum.append(list(map(lambda x, y: x*y, elem, image)))
        weight_images_sum[idx] = reduce(lambda x, y: x+y, weight_images_sum[idx])
    
    # Get the max weigth sum of the list of weight
    predicted_value = weight_images_sum.index(max(weight_images_sum))

    return predicted_value

def main():
    DIGITS = datasets.load_digits()
    DATA = DIGITS['data']
    TARGET = DIGITS['target']

    weight_images = create_weight_images()

    x_train, x_test, y_train, y_test = train_test_split(
        DATA, TARGET, test_size=0.3)

    # Fit the data
    for idx, elem in enumerate(x_train):
        weight_images = calculate_new_weight(elem, y_train[idx], weight_images)

        images_display = weight_images
        max_values = max(map(max, weight_images))
        images_display = [((elem/max_values)*256) for elem in images_display]

        vertical_1 = np.vstack((images_display[0].reshape((8, 8)), images_display[5].reshape((8, 8))))
        vertical_2 = np.vstack((images_display[1].reshape((8, 8)), images_display[6].reshape((8, 8))))
        vertical_3 = np.vstack((images_display[2].reshape((8, 8)), images_display[7].reshape((8, 8))))
        vertical_4 = np.vstack((images_display[3].reshape((8, 8)), images_display[8].reshape((8, 8))))
        vertical_5 = np.vstack((images_display[4].reshape((8, 8)), images_display[9].reshape((8, 8))))

        img = np.hstack((vertical_1, vertical_2, vertical_3,
                        vertical_4, vertical_5))

        images_display = cv2.resize(np.uint8(img), (960, 384), interpolation = cv2.INTER_AREA)

        cv2.imshow('image', np.uint8(images_display))
        
        # Allow to show image point by point
        k = cv2.waitKey(1)

    # Lock it so we can admire the work
    k = cv2.waitKey(99999)

    # Precision calcul of the predictions
    nbr_tests = 0
    correct_prediction = 0

    for idx, elem in enumerate(x_test):
        if predict_right_image(elem, weight_images) == y_test[idx]:
            correct_prediction += 1
        nbr_tests += 1

    print(f"Precision calcul of the predictions : {(correct_prediction/nbr_tests) * 100}%")

if __name__ == "__main__":
    main()