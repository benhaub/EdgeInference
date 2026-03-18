###############################################################################
# File: RunInference.py
# Synopsis: Load an image file and run inference on it using a TFLite model
# Authour: Artificailly generated, with small edits by Ben Haubrich
###############################################################################

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def runInference(imagePath, modelPath):
    
    with open(imagePath, 'rb') as f:
        imageData = f.read()
    
    imageArray = np.frombuffer(imageData, dtype=np.uint8)
    
    imageArray = imageArray.reshape((28, 28))
    
    print(f"Image shape: {imageArray.shape}")
    print(f"Image data range: {imageArray.min()} to {imageArray.max()}")
    
    interpreter = tf.lite.Interpreter(model_path=modelPath)
    interpreter.allocate_tensors()
    
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()
    
    print(f"\nInput details: {inputDetails}")
    print(f"Output details: {outputDetails}")
    
    # The model expects input shape and type based on inputDetails
    inputShape = inputDetails[0]['shape']
    inputType = inputDetails[0]['dtype']
    inputData = imageArray.reshape(inputShape).astype(inputType)

    interpreter.set_tensor(inputDetails[0]['index'], inputData)
    interpreter.invoke()
    outputData = interpreter.get_tensor(outputDetails[0]['index'])
    
    print(f"\nOutput data: {outputData}")
    print(f"Output shape: {outputData.shape}")
    
    predictedDigit = np.argmax(outputData[0])
    confidence = ((outputData[0][predictedDigit] + 128) / 255) * 100
    
    print(f"\nPredicted digit: {predictedDigit}")
    print(f"Confidence: {confidence:.1f}%") # Fixed formatting typo here
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(imageArray, cmap='gray')
    plt.title(f'Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), outputData[0])
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title(f'Inference Results\nPredicted: {predictedDigit}')
    plt.xticks(range(10))
    plt.show()
    
    return predictedDigit, outputData[0]

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run inference on one or multiple images using a TFLite model.")
    
    # Accept 1 or more image paths
    parser.add_argument(
        'images', 
        nargs='+', 
        help="Path(s) to the image file(s) you want to test"
    )
    
    # Optional argument for the model, defaulting to your specified model
    parser.add_argument(
        '-m', '--model', 
        default=None,
        help="Path to the TFLite model file"
    )
 
    args = parser.parse_args()
    modelPath = args.model

    # If no model was explicitly provided, look for one in the current working directory
    if modelPath is None:
        tflite_files = list(Path('.').glob('*.tflite'))
        if not tflite_files:
            print("Error: No .tflite model specified and none found in the current directory.")
            sys.exit(1)
        elif len(tflite_files) > 1:
            print(f"Warning: Multiple .tflite files found. Defaulting to '{tflite_files[0]}'.")
        
        modelPath = str(tflite_files[0])

    # Check if model exists first
    if not Path(modelPath).exists():
        print(f"Error: Model file missing: {modelPath}")
    else:
        # Loop through all provided image paths
        for imagePath in args.images:
            print(f"\n{'='*50}")
            print(f"Processing Image: {imagePath}")
            print(f"{'='*50}")
            
            if Path(imagePath).exists():
                predictedDigit, probabilities = runInference(imagePath, modelPath)
            else:
                print(f"Error: Image file missing: {imagePath}")