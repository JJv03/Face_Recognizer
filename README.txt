FaceRecognizer made following the tutorial of this website:
- https://realpython.com/face-recognition-with-python/ -

Instructions:
* To train the model and generate the .pkl file go to the "training" directory:
Create a new directory with the name of the person to train
Upload as many images as you can of the person in order to get a good model of his traits
Use "python detector.py --train"
* To validate the model:
At the "validation" directory upload the images you want to validate
Use "python detector.py --validate" 
* To test only one image:
Use "python detector.py --test -f _yourImage_"

If you have any doubt of parameters of the program use "python detector.py --help"