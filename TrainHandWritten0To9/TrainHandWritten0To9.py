###############################################################################
#Authour: Ben Haubrich                                                        #
#File: TrainHandWritten0To9.py                                                #
#Date: November 9th, 2025                                                     #
#Synopsis: Trains a Neural Network to recognize handwritten numbers 0 to 9    #
#DataSet: https://www.kaggle.com/datasets/hichamachahboun/mnist-handwritten-digits/code
#References: https://www.kaggle.com/code/tehreemkhan111/mnist-handwritten-digits-ann
#            https://www.tensorflow.org/model_optimization/guide/quantization/training_example?_gl=1*1odfcw6*_up*MQ..*_ga*MjA1NjcwMjUwOS4xNzY1NTgwMDA0*_ga_W0YLR4190T*czE3NjU1ODAwMDQkbzEkZzAkdDE3NjU1ODAxMTQkajYwJGwwJGgw
###############################################################################

# Presently, This Predator G3 has an Nvida GeForce 960, so an older tensorflow library was needed:
# TensoFlow: 2.10.0
# CUDA: 11.2
# CUDNN: 8.1
# conda create -n tf210_cc52 -c conda-forge python=3.10 cudatoolkit=11.2 cudnn=8.1 -y
# conda activate tf210_cc52
# conda install -c conda-forge tensorflow==2.10.0
# conda install matplotlib
# And set LD_LIBRARY_PATH to include $CONDA_PREFIX/lib/
# Use conda activate tf210_cc52 (CC52 for Compute Capability 5.2, which the 960 has) to activate the proper environment
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from fileToArrayOfBytes import fileToArrayOfBytes
from PIL import Image

#Function to display sample images from the set of images and labels given
#Can take variable number of image indeces
def displayImages(imageSet, labelSet, *args):
    num_images = len(args)
    num_cols = min(3, num_images)  # Max 3 images per row
    num_rows = int(np.ceil(num_images / num_cols))

    plt.figure(figsize=(5, 5))

    for i, idx in enumerate(args):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(imageSet[idx], cmap='gray')
        plt.title(f'Label = {labelSet[idx]}', fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plotTrainingHistory(modelFittingHistory):
    train_loss = modelFittingHistory.history['loss']
    val_loss = modelFittingHistory.history['val_loss']
    train_accuracy = modelFittingHistory.history.get('accuracy', None)
    val_accuracy = modelFittingHistory.history.get('val_accuracy', None)
    epochs = range(1, len(train_loss) + 1)

    # Loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    if train_accuracy is not None and val_accuracy is not None:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
    else:
        print("Accuracy data not available in history.")

    plt.tight_layout()
    plt.show()

#Makes the MNIST images look more like the images pulled from the LCD.
def applyLcdDomainAdaptation(images, pixel_size=14):
    processed = []
    for img_array in images:
        # 1. Force to 2D uint8
        img = Image.fromarray(img_array.astype(np.uint8))

        # 2. Hard Threshold: Force every pixel to either 0 or 255
        # This eliminates the "fuzz" immediately.
        img = img.point(lambda p: 255 if p > 127 else 0, mode='1') # mode='1' is bilevel

        # 3. Downsample AND Upsample using NEAREST
        # Do NOT use Bilinear/Bicubic, as they re-introduce blur.
        img_small = img.resize((pixel_size, pixel_size), resample=Image.NEAREST)
        img_blocky = img_small.resize((28, 28), resample=Image.NEAREST)

        # 4. Convert back to 'L' (Grayscale) uint8 for the NN
        processed.append(np.array(img_blocky.convert('L'), dtype=np.uint8))

    return np.array(processed)

#https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/model_optimization?_gl=1*rllyhg*_up*MQ..*_ga*MzA5MTY5ODk1LjE3NjU1Nzk2Mjk.*_ga_P1DBVKWT6V*czE3NjU1Nzk2MjkkbzEkZzAkdDE3NjU1Nzk2MjkkajYwJGwwJGgxNjExNjE4NTE.
def saveQuantizedTfLiteModel(model, xTrainingSetValues, path):
    def representativeDataSet():
        for data in tf.data.Dataset.from_tensor_slices(xTrainingSetValues).batch(1).take(100):
            dataWithChannel = tf.expand_dims(data, axis=-1)
            yield [tf.dtypes.cast(dataWithChannel, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representativeDataSet
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.int8
    tflite_quantized_model = converter.convert()

    with open(path, 'wb') as f:
        f.write(tflite_quantized_model)

if __name__ == '__main__':
    tf.keras.backend.clear_session()

    xTrainingImageSet = 'DataSetZeroToNine/train_images.npy'
    yTrainingLabelSet = 'DataSetZeroToNine/train_labels.npy'
    xTestingImageSet  = 'DataSetZeroToNine/test_images.npy'
    yTestingLabelSet  = 'DataSetZeroToNine/test_labels.npy'
    xTrainingImages = applyLcdDomainAdaptation(np.load(xTrainingImageSet))
    yTrainingLabels = np.load(yTrainingLabelSet)
    xTestingImages  = applyLcdDomainAdaptation(np.load(xTestingImageSet))
    yTestingLabels  = np.load(yTestingLabelSet)

    #display labels 0 though 9
    #displayImages(xTrainingImages, yTrainingLabels, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    #Define the training model. Use a basic sequential model where the input is fed forward through each layer of the NN
    #and has one output for each number 0 through 9.
    #TODO: Use sparse? https://www.tensorflow.org/guide/sparse_tensor
    model = tf.keras.models.Sequential([
        #Input layer
        tf.keras.layers.Input(shape=(28,28, 1)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        #4 Hidden layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        #Output layer. Softmax takes in an input vector of the outputs of the neural network and produces a probability that
        #the output belongs to one of the labels in the data set. It's commonly used as the output of a network and is good
        #for multiple classification problems (more than 2 labels)
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()

    #The loss function is chosen based on how you label your data. These labels are integer encoded, so this is the appropriate
    #loss function. Labels could also be 1-hot-encoded [0,0,0,1,0,0,0,0,0,0] to represent the number 3 in which case 
    #Categroial cross-entropy would be appropriate.
    model.compile(loss = "sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=["accuracy"])

    history = model.fit(xTrainingImages, yTrainingLabels,
                        epochs=5,
                        validation_data = (xTestingImages, yTestingLabels), batch_size=128)

    plotTrainingHistory(history)

    saveQuantizedTfLiteModel(model, xTrainingImages, "quantizedHandwrittenZeroToNineModel.tflite")

    fileToArrayOfBytes("quantizedHandwrittenZeroToNineModel.tflite", ".")