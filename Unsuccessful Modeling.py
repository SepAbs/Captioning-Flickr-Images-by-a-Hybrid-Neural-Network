from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.utils import to_categorical
from matplotlib.pyplot import figure, imshow, ion, legend, plot, savefig, show, subplots, title, xlabel, ylabel
from numpy import argmax, asarray, dstack, mean, sqrt, unique
from os import environ
from pandas import read_csv
from seaborn import histplot
#from pytorch2keras import pytorch_to_keras
from torchvision.models import resnet18
from warnings import filterwarnings
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
import torch
filterwarnings("ignore")
environ["KMP_DUPLICATE_LIB_OK"], environ["TF_CPP_MIN_LOG_LEVEL"], environ["TF_ENABLE_ONEDNN_OPTS"] = "TRUE", "3", "0"

# A class for plotting evaluation of model's performance on test set epoch by epoch
class TestCallback(Callback):
    def __init__(self):
        self.loss, self.acc = [], []

    def on_epoch_end(self, epoch, logs = {}):
        Loss, accuracy = self.model.evaluate(X_test, caty_test, verbose = OFF)
        self.loss.append(Loss)
        self.acc.append(accuracy)
        # print(f"Testing loss: {Loss}, Accuracy: {accuracy}")
"""
# Loading & normalizing the dataset
trainSet, testSet, inputLayer, numberFilters, filterSize, ReLU, Softmax, poolSize, Strides, lossFunction, Accuracy, Loss, validationAccuracy, validationLoss, Epochs, Title, Alpha, localPopulation, Numbers, numberIterations, numberGenerations, Fitness, Parameters, Optimizer, Legends, Location, XLabel, YLabel, Blue, Green, Red, figureSize, DPI, train_batch_size, test_batch_size, validationSplit, OFF, Axis, Tuner = read_csv("mnist_train.csv"), read_csv("mnist_test.csv"), Input(shape = (48, 48, 3)), [2, 4, 6, 8, 10], (3, 3), "relu", "softmax", (2, 2), 2, "categorical_crossentropy", "accuracy", "loss", "val_accuracy", "val_loss", 15, "Convolutional Neural Network Evaluation", 0.6, 1, 10, 1, 1, [], [], "adam", ["Train Loss", "Validation Loss", "Test Loss"], "upper right", "Epochs", "Loss", "blue", "green", "red", (13, 5), 1200, 1000, 100, 0.2, 0, 1, {}
df, Length, rangeEpochs, Metric, XAcc = trainSet._append(testSet, ignore_index = True), int(sqrt(trainSet.shape[1])), range(Epochs), [Accuracy], Accuracy.title()
X_train, X_test, y_train, y_test = trainSet.drop(["label"], axis = 1).astype("float32") / 255., testSet.drop(["label"], axis = 1).astype("float32") / 255., trainSet["label"].astype("int"), testSet["label"].astype("int")
dfX_train, localBound, number_train_samples, number_test_samples, numberClasses,  = X_train, X_train.shape[1], len(X_train) // 2, len(X_test), len(unique(y_test)), 
"""

def ResNet18(input_shape):
    """
    Function to create a ResNet-18 model using Keras.
 
    Parameters:
    - input_shape: tuple
        The shape of the input images in the format (height, width, channels).
    - num_classes: int
        The number of output classes for classification.
 
    Returns:
    - keras.Model:
        The ResNet-18 model.
 
    Raises:
    - ValueError:
        Will raise an error if the input_shape is invalid or the num_classes is less than 2.
    """
 
    # Checking if the input_shape is valid
    if len(input_shape) != 3 or input_shape[0] <= 0 or input_shape[1] <= 0 or input_shape[2] <= 0:
        raise ValueError("Invalid input_shape. Please provide a valid shape (height, width, channels).")

    inputLayer = Input(shape = input_shape)

    # Residual blocks
    Residual = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputLayer))))
 
    for Layers in range(2):
        Residual = Activation('relu')(BatchNormalization()(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation("relu")(BatchNormalization()(Conv2D(64, kernel_size = (3, 3), strides = (1, 1), padding = "same")(Residual))))) + Residual)
 
    # Residual blocks
    Residual = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(BatchNormalization()(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same')(Residual))))))

    for Layer in range(2):
        Residual = Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(Residual))))) + Residual)
 
    # Residual blocks
    Residual = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(BatchNormalization()(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same')(Residual))))))
    for Layers in range(2):
        Residual = Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(Residual))))) + Residual)
 
    # Residual blocks
    Residual = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(BatchNormalization()(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(Residual))))))
                                                                                                   
    for Layers in range(2):
        Residual = Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(Activation('relu')(BatchNormalization()(Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(Residual))))) + Residual)

    # Create the model
    return Model(inputs = inputLayer, outputs = Dense(1000, activation='softmax')(Dense(512, activation='relu')(Flatten()(Residual))))


# Example usage:

Model = ResNet18((256, 256, 3))
Model.summary()

ResNet18 = resnet18()
ResNet18.load_state_dict(torch.load("ResNet18 Pretrained Weights.pth"))
ResNet18.eval()
layerWeights = zip(Model.layers, ResNet18.state_dict())

# Iterate over the layers in the Keras model and the weights in the PyTorch model
for Layer, Weights in layerWeights:
    # Set the weights of the corresponding layers in the Keras model
    Layer.set_weights(Weights)

print(Model.summary())
