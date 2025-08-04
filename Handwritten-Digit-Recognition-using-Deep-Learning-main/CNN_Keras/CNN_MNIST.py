import numpy as np
import argparse
import cv2
from cnn.neural_network import CNN
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# ------------------ Parse CLI Arguments ------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--save_weights", type=str, default="cnn_weights.hdf5")
args = vars(ap.parse_args())

# ------------------ Load MNIST Dataset ------------------ #
print('Loading MNIST Dataset...')
dataset = fetch_openml('mnist_784', version=1, as_frame=False)

mnist_data = dataset.data.reshape((dataset.data.shape[0], 28, 28)).astype("float32") / 255.0
mnist_data = mnist_data[:, np.newaxis, :, :]
labels = dataset.target.astype("int")

# Split into training and testing sets
train_img, test_img, train_labels, test_labels = train_test_split(
    mnist_data, labels, test_size=0.1, random_state=42
)

# One-hot encode labels
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# ------------------ Build and Compile Model ------------------ #
print('\nCompiling model...')
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

clf = CNN.build(
    width=28,
    height=28,
    depth=1,
    total_classes=10,
    Saved_Weights_Path=args["save_weights"] if args["load_model"] > 0 else None
)
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# ------------------ Train or Load Model ------------------ #
if args["load_model"] < 0:
    print('\nTraining the Model...')
    clf.fit(train_img, train_labels, batch_size=128, epochs=20, verbose=1)

    print('Evaluating Accuracy and Loss Function...')
    loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
    print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

# ------------------ Save Model Weights ------------------ #
if args["save_model"] > 0:
    print('Saving weights to file...')
    clf.save_weights(args["save_weights"], overwrite=True)

# ------------------ Predict and Display Sample Results ------------------ #
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
    probs = clf.predict(test_img[np.newaxis, num])
    prediction = probs.argmax(axis=1)

    image = (test_img[num][0] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    print('Predicted Label: {}, Actual Value: {}'.format(prediction[0], np.argmax(test_labels[num])))
    # Uncomment to view image
    # cv2.imshow('Digit', image)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
