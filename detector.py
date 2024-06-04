import argparse
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw
import face_recognition
import pickle

# Constant variable for the path of output information
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories that don't exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

# Definition of the parser 
parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--validate", action="store_true", help="Validate trained model")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument(
    "-m",
    action="store",
    default="hog",
    choices=["hog", "cnn"],
    help="Which model to use for training: hog (CPU), cnn (GPU)",
)
parser.add_argument(
    "-f", action="store", help="Path to an image with an unknown face"
)
args = parser.parse_args()

# Loads all the training images at "training" directory, finds the faces within the images, 
# and then creates a dictionary containing the two lists that it created with each image.
# Then saved that dictionary to disk so that you could reuse the encodings.
# This is for KNOWN faces
def encode_known_faces(
    # hog (histogram of oriented gradients):    it works best with a CPU
    # cnn (convolutional neural network):       it works better on a GPU
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    
    # Goes through each directory in the "training" directory
    for filepath in Path("training").glob("*/*"):
        # Saves the name of the directory in the variable "name"
        name = filepath.parent.name
        # Saves the information of the image in the variable "image"
        image = face_recognition.load_image_file(filepath)
        
        # Detects the locations of faces in each image (return four coordinates of a box)
        face_locations = face_recognition.face_locations(image, model=model)
        # Generates encodings for the detected faces in an image (return encodings)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Adds encodings and name to the list
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
            
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

# Check if the passed image contains the face of one of the registered faces
# This is for NOT KNOWN faces
def recognize_faces(
    image_location: str,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    # Loads the encodings from the "output" directory
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Saves the information of the image in the variable "image"
    input_image = face_recognition.load_image_file(image_location)
    
    # Detects the locations of faces in the image (return four coordinates of a box)
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    # Generates encodings for the detected faces in the image (return encodings)
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    
    # Prepares the draw variable to print the recognized faces
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    
    # Iterates through the faces at the image
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        # Searchs for a name that corresponds to one already registered
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        # Displays the image with the detected face and his name
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()
        
# Returns the name of a registered person if one of the saved encodings correspond to the unknown one
def _recognize_face(unknown_encoding, loaded_encodings):
    # Compare all the registered enconding with the unknown one (return list of True/False)
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    # Search for the amount of votes of each known face
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    # Returns the most voted
    if votes:
        return votes.most_common(1)[0][0]
    
# Displays the bounding box of a face with his name
def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

# Uses the function "recognize_faces" on all the "validation" directory images
def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()), model=model
            )

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.test:
        recognize_faces(image_location=args.f, model=args.m)