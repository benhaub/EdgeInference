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

#https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/model_optimization?_gl=1*rllyhg*_up*MQ..*_ga*MzA5MTY5ODk1LjE3NjU1Nzk2Mjk.*_ga_P1DBVKWT6V*czE3NjU1Nzk2MjkkbzEkZzAkdDE3NjU1Nzk2MjkkajYwJGwwJGgxNjExNjE4NTE.
def saveQuantizedTfLiteModel(model, xTrainingSetValues, path):
    def representativeDataSet():
            # 1. Create dataset and add the channel dimension (28, 28) -> (28, 28, 1)
            ds = tf.data.Dataset.from_tensor_slices(
                tf.expand_dims(xTrainingSetValues, axis=-1)
            )
            
            # 2. Resize to match model input and batch by 1 to get the 4th dimension
            # Resulting shape per 'data' will be (1, 120, 160, 1)
            ds = ds.map(lambda x: tf.image.resize(x, (120, 160)))
            ds = ds.batch(1).take(100)

            for data in ds:
                # data is already (1, 120, 160, 1) and float32 from resize
                yield [data]

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
    xTrainingImages = np.load(xTrainingImageSet)
    yTrainingLabels = np.load(yTrainingLabelSet)
    xTestingImages = np.load(xTestingImageSet)
    yTestingLabels = np.load(yTestingLabelSet)

    #Define the training model. Use a basic sequential model where the input is fed forward through each layer of the NN
    #and has one output at the end.
    lcdScreenSize = (120, 160, 1)
    model = tf.keras.models.Sequential([
        #Input layer
        tf.keras.layers.Resizing(28,28, input_shape=lcdScreenSize),
        #Put rescaling in the model so that we don't have to normalize the data before giving it to the model on the device.
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, (3, 3) , activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3) , activation='relu'),
        #The input shape must be flattened for the hidden layers inside the neural network to process. The input is a 28x28,
        #but the dense layers need a 1D vector.
        tf.keras.layers.GlobalAveragePooling2D(),
        #4 Hidden layers
        tf.keras.layers.Dense(32, activation='relu'),
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
    
    def resizeMnistToLcd(image, label):
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, (120, 160))
        return image, label

    trainImagesToLcdSize = tf.data.Dataset.from_tensor_slices(
        (xTrainingImages, yTrainingLabels)
    ).map(resizeMnistToLcd, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(16)\
    .prefetch(tf.data.AUTOTUNE)

    testImagesToLcdSize = tf.data.Dataset.from_tensor_slices(
        (xTestingImages, yTestingLabels)
    ).map(resizeMnistToLcd, num_parallel_calls=tf.data.AUTOTUNE)\
    .batch(16)\
    .prefetch(tf.data.AUTOTUNE)
    
    history = model.fit(trainImagesToLcdSize,
                        epochs=10,
                        validation_data = testImagesToLcdSize)

    plotTrainingHistory(history)

    saveQuantizedTfLiteModel(model, xTrainingImages, "quantizedHandwrittenZeroToNineModel.tflite")

    fileToArrayOfBytes("quantizedHandwrittenZeroToNineModel.tflite", ".")