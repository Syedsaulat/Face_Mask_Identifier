# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def parse_arguments():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output loss/accuracy plot")
    ap.add_argument("-m", "--model", type=str,
        default="mask_detector.model",
        help="path to output face mask detector model")
    return vars(ap.parse_args())

def load_and_preprocess_data(dataset_path):
    """Load and preprocess images from the dataset."""
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the input image (224x224) and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(label)

    # convert to numpy arrays and perform one-hot encoding
    data = np.array(data, dtype="float32")
    labels = np.array(labels)
    
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    return train_test_split(data, labels,
        test_size=0.20, stratify=labels, random_state=42), lb

def create_model():
    """Create and compile the face mask detector model."""
    # load the MobileNetV2 network
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

    # construct the enhanced head of the model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(256, activation="relu")(headModel)
    headModel = BatchNormalization()(headModel)
    headModel = Dropout(0.3)(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)

    # create the final model
    model = Model(inputs=baseModel.input, outputs=headModel)

    # freeze base model layers
    for layer in baseModel.layers:
        layer.trainable = False

    return model

def train_model(model, trainX, trainY, testX, testY, args):
    """Train the model and plot results."""
    # initialize hyperparameters
    INIT_LR = 1e-4
    EPOCHS = 2
    BS = 32

    # initialize the training data augmentation object
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    # compile model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the model
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    return H

def evaluate_model(model, testX, testY, lb, BS):
    """Evaluate the model performance."""
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)
    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=lb.classes_))

def plot_training_history(H, args):
    """Plot training history."""
    N = len(H.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])

def main():
    """Main function to run the training pipeline."""
    # parse command line arguments
    args = parse_arguments()

    # load and preprocess data
    (trainX, testX, trainY, testY), lb = load_and_preprocess_data(args["dataset"])

    # create and compile model
    model = create_model()

    # train model
    H = train_model(model, trainX, trainY, testX, testY, args)

    # evaluate model
    evaluate_model(model, testX, testY, lb, BS=32)

    # save the model to disk
    print("[INFO] saving mask detector model...")
    if args["model"].endswith(".h5"):
        model.save(args["model"])
    else:
        model.save(args["model"] + ".h5")

    # plot training history
    plot_training_history(H, args)

if __name__ == "__main__":
    main()