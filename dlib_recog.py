import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance

# Khởi tạo dlib face detector và facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Cần tải file này
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Cần tải file này

# Thư mục chứa ảnh database
database_path = "dataset/"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Hàm để tính face descriptor từ ảnh
def get_face_descriptor(image_path):
    img = cv2.imread(image_path)
    dets = detector(img, 1)
    if len(dets) > 0:
        shape = predictor(img, dets[0])
        face_descriptor = face_rec_model.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)
    return None

# Load database descriptors
database_descriptors = {}
for root, dirs, files in os.walk(database_path):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(root, file)
            descriptor = get_face_descriptor(img_path)
            if descriptor is not None:
                name = os.path.basename(root)
                database_descriptors[img_path] = (name, descriptor)

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Phát hiện khuôn mặt trong frame
        dets = detector(frame, 1)
        
        if len(dets) > 0:
            # Lấy descriptor cho khuôn mặt đầu tiên được phát hiện
            shape = predictor(frame, dets[0])
            current_descriptor = np.array(face_rec_model.compute_face_descriptor(frame, shape))
            
            # So sánh với database
            min_dist = float('inf')
            match_name = "No match found"
            match_path = None
            
            for path, (name, db_descriptor) in database_descriptors.items():
                dist = distance.euclidean(current_descriptor, db_descriptor)
                if dist < min_dist and dist < 0.6:  # Ngưỡng 0.6 thường được sử dụng với dlib
                    min_dist = dist
                    match_name = name
                    match_path = path
            
            text = f"Match: {match_name}"
            
            # Vẽ rectangle quanh khuôn mặt
            d = dets[0]
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
        else:
            text = "No match found"
        
        # Vẽ text lên frame
        cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception as e:
        print("Lỗi nhận diện:", e)
    
    # Hiển thị hình ảnh
    cv2.imshow('Dlib Face Recognition', frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()