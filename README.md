# FaceRecognizer

This project is a face recognition system built following the tutorial from [Real Python](https://realpython.com/face-recognition-with-python/).

## Instructions

### Training the Model

1. Navigate to the `training` directory.
2. Create a new directory named after the person you want to train the model on.
3. Upload as many images of the person as possible to achieve a good model of their features.
4. Run the training script:

    python detector.py --train

### Validating the Model

1. Navigate to the `validation` directory.
2. Upload the images you want to use for validation.
3. Run the validation script:

    python detector.py --validate

### Testing a Single Image

To test the model with a single image, run: 
  
    python detector.py --test -f yourImage

### Help

For a list of available parameters and their descriptions, use: 
   
    python detector.py --help

If you have any questions or need further assistance, please refer to the help command above.
