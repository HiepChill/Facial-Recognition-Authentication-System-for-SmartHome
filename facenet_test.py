import cv2
import os
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Thư mục chứa ảnh database
database_path = "dataset/"
if not os.path.exists(database_path):
    os.makedirs(database_path)

# Khởi tạo mô hình nhận diện khuôn mặt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.9],
    factor=0.709,
    post_process=True,
    device=device
)
model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Hàm nạp dữ liệu khuôn mặt từ database
def load_face_database():
    face_db = {}
    for person_name in os.listdir(database_path):
        person_folder = os.path.join(database_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Lỗi đọc ảnh: {image_path}, bỏ qua...")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_tensor = mtcnn(img_rgb)
                
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    embedding = model(face_tensor).detach().cpu().numpy().flatten()
                    face_db[person_name] = embedding
    return face_db

# Nạp database khuôn mặt
face_database = load_face_database()

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    try:
        # Chuyển sang RGB cho MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Phát hiện khuôn mặt
        boxes, probs = mtcnn.detect(frame_rgb)
        
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                if probs[i] < 0.9:  # Lọc khuôn mặt có độ tin cậy thấp
                    continue
                    
                # Lấy tọa độ khuôn mặt
                left, top, right, bottom = [int(b) for b in box]
                
                # Xử lý khuôn mặt
                face_img = frame_rgb[top:bottom, left:right]
                face_tensor = mtcnn(face_img)
                
                if face_tensor is not None:
                    # Tính embedding
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    face_embedding = model(face_tensor).detach().cpu().numpy().flatten()
                    
                    # Nhận diện khuôn mặt
                    match_name = "No match found"
                    max_similarity = -1
                    
                    # So sánh với database bằng cosine similarity
                    for name, db_embedding in face_database.items():
                        similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match_name = name if max_similarity > 0.5 else "Unknown"  # Ngưỡng 0.5
                    
                    # Vẽ khung nhận diện
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Hiển thị tên trên ảnh
                    cv2.putText(frame, match_name, (left, max(top - 10, 20)), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception as e:
        print("Lỗi nhận diện:", e)
    
    # Hiển thị hình ảnh
    cv2.imshow('FaceNet Recognition', frame)
    
    # Thoát nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()