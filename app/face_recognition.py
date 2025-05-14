import cv2
import numpy as np
import os
import uuid
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from .config import FACE_RECOGNITION_THRESHOLD, TEMP_DIR, NOTIFICATION_INTERVAL_KNOWN, NOTIFICATION_INTERVAL_UNKNOWN
from .database import log_detection
from datetime import datetime
import threading
import time

PROVIDER_LIST = [
    'DmlExecutionProvider',
    'CUDAExecutionProvider',
    'CPUExecutionProvider'
]

recognition_lock = threading.Lock()
last_notification_times = {}  # {user_key: timestamp}

def initialize_face_analyzer():
    """Khởi tạo FaceAnalysis với các provider khả dụng"""
    for provider in PROVIDER_LIST:
        try:
            print(f"Đang thử khởi tạo FaceAnalysis với {provider}...")
            face_analyzer = FaceAnalysis(name='buffalo_l', providers=[provider])
            face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print(f"Khởi tạo thành công với {provider}")
            return face_analyzer
        except Exception as e:
            print(f"Khởi tạo thất bại với {provider}: {str(e)}")
    raise RuntimeError("Không thể khởi tạo FaceAnalysis với bất kỳ provider nào")

face_analyzer = initialize_face_analyzer()
face_database = {}

def load_face_database(user_face_data):
    """Tải dữ liệu khuôn mặt từ cơ sở dữ liệu vào bộ nhớ"""
    face_db = {}
    user_info = {}
    for user_id, name, image_path in user_face_data:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            faces = face_analyzer.get(img)
            if faces:
                embedding = faces[0].normed_embedding
                user_key = f"{user_id}_{name}"
                if user_key not in face_db:
                    face_db[user_key] = []
                    user_info[user_key] = {"name": name, "user_id": user_id}
                face_db[user_key].append(embedding)
    averaged_db = {}
    for user_key, embeddings in face_db.items():
        if embeddings:
            averaged_embedding = np.mean(embeddings, axis=0)
            averaged_db[user_key] = {
                "embedding": averaged_embedding,
                "info": user_info[user_key]
            }
    return averaged_db

def save_frame(frame):
    """Lưu frame vào thư mục tạm thời"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = uuid.uuid4().hex[:8]
    filename = f"detection_{timestamp}_{random_id}.jpg"
    filepath = os.path.join(TEMP_DIR, filename)
    success = cv2.imwrite(filepath, frame)
    if success:
        return filepath
    return None

def process_frame(frame, face_database):
    """Xử lý frame để nhận diện khuôn mặt"""
    display_frame = frame.copy()
    recognized_users = []
    try:
        faces = face_analyzer.get(frame)
        for face in faces:
            face_embedding = face.normed_embedding
            match_info, max_similarity = recognize_face(face_embedding, face_database)
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            draw_recognition_result(display_frame, left, top, right, bottom, 
                                   match_info, max_similarity, recognized_users)
            
            current_time = time.time()
            user_key = f"{match_info['user_id']}_{match_info['name']}" if match_info else "unknown"
            last_sent = last_notification_times.get(user_key, 0)
            interval = NOTIFICATION_INTERVAL_KNOWN if match_info else NOTIFICATION_INTERVAL_UNKNOWN
            
            if current_time - last_sent >= interval:
                temp_path = save_frame(frame)
                if match_info:
                    with recognition_lock:
                        threading.Thread(
                            
                            args=(match_info, max_similarity, temp_path)
                        ).start()
                else:
                    with recognition_lock:
                        threading.Thread(
                            
                            args=(temp_path,)
                        ).start()
                last_notification_times[user_key] = current_time
    except Exception as e:
        print(f"Lỗi nhận diện: {e}")
    return display_frame, recognized_users


def recognize_face(face_embedding, face_database):
    """Nhận diện khuôn mặt bằng cách so sánh embedding"""
    match_info = None
    max_similarity = -1
    for user_key, data in face_database.items():
        db_embedding = data["embedding"]
        similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > FACE_RECOGNITION_THRESHOLD:
                match_info = data["info"]
            else:
                match_info = None
    return match_info, max_similarity

def draw_recognition_result(frame, left, top, right, bottom, match_info, similarity, recognized_users):
    """Vẽ kết quả nhận diện lên frame"""
    if match_info:
        color = (0, 255, 0)
        name = match_info["name"]
        user_id = match_info["user_id"]
        label = f"{name} ({similarity:.2f})"
        recognized_users.append({
            "user_id": user_id,
            "name": name,
            "confidence": float(similarity)
        })
    else:
        color = (0, 0, 255)
        label = f"Unknown ({similarity:.2f})"
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    y_position = max(top - 10, 20)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, 
                 (left, y_position - text_size[1] - 5), 
                 (left + text_size[0], y_position + 5), 
                 (0, 0, 0), 
                 -1)
    cv2.putText(frame, label, (left, y_position),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)