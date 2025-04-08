""" Để lưu encodings và tên của các faces trong dataset vào file encodings.pickle"""
# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

from imutils import paths
import argparse
import pickle
import cv2
import os
import face_recognition
import imutils
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-d", "--detection_method", type=str, default="hog", help="face detector to use: cnn or hog")
args = vars(ap.parse_args())

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

def process_image(imagePath):
    # Lấy tên người từ đường dẫn
    name = imagePath.split(os.path.sep)[-2]

    # Load và resize ảnh để tăng tốc
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    if len(boxes) == 0:
        print(f"[WARNING] No face found in {imagePath}. Skipping...")
        return None

    # Encode face
    encodings = face_recognition.face_encodings(rgb, boxes)
    return [(encoding, name) for encoding in encodings]

# Sử dụng đa luồng để xử lý ảnh song song
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, imagePaths))

# Lưu các encodings và tên
for result in results:
    if result is not None:
        for (encoding, name) in result:
            knownEncodings.append(encoding)
            knownNames.append(name)

# Lưu dữ liệu vào file pickle
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encoding completed successfully!")



    







