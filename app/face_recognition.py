import cv2
import numpy as np
import os
import uuid
import insightface
from sklearn.metrics.pairwise import cosine_similarity
from .config import FACE_RECOGNITION_THRESHOLD, TEMP_DIR, NOTIFICATION_INTERVAL_KNOWN, NOTIFICATION_INTERVAL_UNKNOWN
from .database import log_detection
from datetime import datetime
import threading
import time
from .telegram_notification import notify_recognized_person, notify_unknown_person

recognition_lock = threading.Lock()
last_notification_times = {}  # {user_key: timestamp}

def initialize_face_analyzer():
    """Khởi tạo InsightFace với GPU nếu có"""
    try:
        # Khởi tạo InsightFace app
        app = insightface.app.FaceAnalysis(
            providers=['DmlExecutionProvider', 'CPUExecutionProvider', 'CUDAExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Khởi tạo InsightFace thành công")
        return app
    except Exception as e:
        print(f"Lỗi khởi tạo InsightFace: {str(e)}")
        raise RuntimeError("Không thể khởi tạo InsightFace")

face_app = initialize_face_analyzer()
face_database = {}

def load_face_database(user_face_data):
    """Tải dữ liệu khuôn mặt từ cơ sở dữ liệu vào bộ nhớ"""
    face_db = {}
    user_info = {}
    
    for user_id, name, image_path in user_face_data:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                # InsightFace sử dụng BGR format trực tiếp
                faces = face_app.get(img)
                
                if faces and len(faces) > 0:
                    # Sử dụng khuôn mặt có diện tích lớn nhất nếu phát hiện nhiều khuôn mặt
                    max_area = 0
                    best_face = None
                    for face in faces:
                        bbox = face.bbox
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_area = area
                            best_face = face
                    
                    if best_face is not None:
                        embedding = best_face.embedding
                        
                        user_key = f"{user_id}_{name}"
                        if user_key not in face_db:
                            face_db[user_key] = []
                            user_info[user_key] = {"name": name, "user_id": user_id}
                        face_db[user_key].append(embedding)

    # Lọc và tính trung bình các embedding
    averaged_db = {}
    for user_key, embeddings in face_db.items():
        if embeddings:
            # Nếu có nhiều embedding, thực hiện lọc để loại bỏ ngoại lai
            if len(embeddings) > 1:
                # Tính trung bình các embedding
                avg_embedding = np.mean(embeddings, axis=0)
                # Tính độ tương đồng của mỗi embedding với trung bình
                similarities = []
                for emb in embeddings:
                    sim = cosine_similarity([emb], [avg_embedding])[0][0]
                    similarities.append(sim)
                # Chọn ra các embedding có độ tương đồng cao
                good_embeddings = [
                    embeddings[i] for i in range(len(embeddings)) 
                    if similarities[i] > 0.8  # Lọc những embedding khác biệt
                ]
                if good_embeddings:
                    averaged_embedding = np.mean(good_embeddings, axis=0)
                else:
                    averaged_embedding = avg_embedding
            else:
                averaged_embedding = embeddings[0]
                
            averaged_db[user_key] = {
                "embedding": averaged_embedding,
                "info": user_info[user_key]
            }
    
    print(f"Đã tải {len(averaged_db)} người dùng vào cơ sở dữ liệu nhận diện khuôn mặt")
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
        # InsightFace xử lý trực tiếp với BGR format
        faces = face_app.get(frame)
        
        if not faces or len(faces) == 0:
            return display_frame, recognized_users
        
        for face in faces:
            try:
                # Lấy thông tin khuôn mặt
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                x1, y1, x2, y2 = bbox
                
                # Nhận diện khuôn mặt
                match_info, similarity = recognize_face(embedding, face_database)
                
                # Vẽ kết quả
                draw_recognition_result(display_frame, x1, y1, x2, y2, 
                                      match_info, similarity, recognized_users)
                
                # Xử lý thông báo
                current_time = time.time()
                user_key = f"{match_info['user_id']}_{match_info['name']}" if match_info else "unknown"
                last_sent = last_notification_times.get(user_key, 0)
                interval = NOTIFICATION_INTERVAL_KNOWN if match_info else NOTIFICATION_INTERVAL_UNKNOWN
                
                if current_time - last_sent >= interval:
                    temp_path = save_frame(frame)
                    if match_info:
                        with recognition_lock:
                            threading.Thread(
                                target=handle_known_person,
                                args=(match_info, similarity, temp_path)
                            ).start()
                    else:
                        with recognition_lock:
                            threading.Thread(
                                target=handle_unknown_person,
                                args=(temp_path,)
                            ).start()
                    last_notification_times[user_key] = current_time
            except Exception as e:
                print(f"Lỗi xử lý khuôn mặt: {e}")
                continue
    except Exception as e:
        print(f"Lỗi nhận diện: {e}")
    
    return display_frame, recognized_users

def handle_known_person(match_info, similarity, image_path):
    """Xử lý trường hợp nhận diện người quen"""
    user_id = match_info["user_id"]
    name = match_info["name"]
    # Đảm bảo similarity là Python float, không phải numpy scalar
    similarity_float = float(similarity)
    notification_sent = notify_recognized_person(name, user_id, similarity_float, image_path)
    log_detection(user_id, name, True, similarity_float, image_path, notification_sent)

def handle_unknown_person(image_path):
    """Xử lý trường hợp nhận diện người lạ"""
    notification_sent = notify_unknown_person(image_path)
    log_detection("unknown", "Unknown Person", False, 0.0, image_path, notification_sent)

def recognize_face(face_embedding, face_database):
    """Nhận diện khuôn mặt bằng cách so sánh embedding"""
    match_info = None
    max_similarity = -1
    
    if len(face_database) == 0:
        return None, 0
        
    # Danh sách chứa các kết quả
    similarities = []
    
    for user_key, data in face_database.items():
        try:
            db_embedding = data["embedding"]
            
            # Đảm bảo các đặc trưng là mảng 1 chiều
            if isinstance(face_embedding, np.ndarray) and isinstance(db_embedding, np.ndarray):
                # định hình lại nếu cần
                if face_embedding.ndim > 1:
                    face_embedding = face_embedding.flatten()
                if db_embedding.ndim > 1:
                    db_embedding = db_embedding.flatten()
                
                # Tính toán độ tương đồng cosine
                similarity = cosine_similarity([face_embedding], [db_embedding])[0][0]
                similarities.append((similarity, data["info"]))
        except Exception as e:
            print(f"Lỗi khi so sánh với người dùng {user_key}: {e}")
            continue
    
    # Sắp xếp theo độ tương đồng
    similarities.sort(reverse=True)
    
    # Điều kiện tối ưu: nếu kết quả tốt nhất lớn hơn ngưỡng và 
    # cách biệt với kết quả thứ 2 đủ lớn (tăng độ tin cậy)
    if similarities and similarities[0][0] > FACE_RECOGNITION_THRESHOLD:
        max_similarity = similarities[0][0]
        match_info = similarities[0][1]
        
        # Nếu có kết quả thứ 2 và kết quả thứ nhất không vượt trội
        if len(similarities) > 1 and max_similarity - similarities[1][0] < 0.1:
            # Nếu sự khác biệt quá nhỏ, đánh dấu là không nhận diện được
            # Điều này giúp tránh nhận diện chập chờn giữa các người dùng tương tự
            if max_similarity < 0.7:  # Ngưỡng tin cậy cao
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