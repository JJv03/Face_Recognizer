# FaceRecognizer

This project is a face recognition system built following the tutorials from [Real Python](https://realpython.com/face-recognition-with-python/) and [AutoDesk Instructables](https://www.instructables.com/Real-time-Face-Recognition-an-End-to-end-Project/).

## Instructions

### To use the program

Install the dependencies of the program

    python -m pip install -r requirements.txt

### Training the Model

1. Navigate to the `training` directory.
2. Create a new directory named after the person you want to train the model on.
3. Upload as many images of the person as possible to achieve a good model of their features.
4. Run the training script:

    ```bash
    python detector.py --train
    ```

### Validating the Model

1. Navigate to the `validation` directory.
2. Upload the images you want to use for validation.
3. Run the validation script:

    ```bash
    python detector.py --validate
    ```

### Testing a Single Image

To test the model with a single image, run: 

    python detector.py --test -f "yourImage"

### Real Time Recognizer

With a model trained, run: 

    python detector.py --realTime

### Help

For a list of available parameters and their descriptions, use: 
   
    python detector.py --help

If you have any questions or need further assistance, please refer to the help command above.
