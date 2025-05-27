import cv2
import numpy as np
import os
import uuid
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
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
    """Khởi tạo FaceNet với GPU nếu có"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Tăng margin để cắt nhiều hơn quanh khuôn mặt
        mtcnn = MTCNN(
            image_size=160,  # Giữ kích thước 160x160 cho chất lượng tốt
            margin=20,       # Tăng margin để có nhiều thông tin quanh khuôn mặt
            min_face_size=40,  # Đặt kích thước mặt tối thiểu
            thresholds=[0.6, 0.7, 0.9],  # Tăng ngưỡng cho độ chính xác cao hơn
            factor=0.709,     # Scaling factor tối ưu
            post_process=True,  # Thực hiện xử lý sau phát hiện
            device=device
        )
        # Sử dụng model vggface2 (cân bằng giữa tốc độ và chính xác)
        model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        print(f"Khởi tạo FaceNet thành công (sử dụng {device})")
        return mtcnn, model
    except Exception as e:
        print(f"Lỗi khởi tạo FaceNet: {str(e)}")
        raise RuntimeError("Không thể khởi tạo FaceNet")

mtcnn, face_model = initialize_face_analyzer()
face_database = {}

def load_face_database(user_face_data):
    """Tải dữ liệu khuôn mặt từ cơ sở dữ liệu vào bộ nhớ"""
    face_db = {}
    user_info = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for user_id, name, image_path in user_face_data:
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Phát hiện khuôn mặt trước để đảm bảo chất lượng
                boxes, probs = mtcnn.detect(img_rgb)
                if boxes is not None and len(boxes) > 0:
                    # Sử dụng khuôn mặt có diện tích lớn nhất nếu phát hiện nhiều khuôn mặt
                    max_area = 0
                    max_box_idx = 0
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            max_box_idx = i
                    
                    # Cắt khuôn mặt và sử dụng MTCNN để xử lý trực tiếp
                    box = boxes[max_box_idx]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    face_img = img_rgb[y1:y2, x1:x2]
                    
                    # Sử dụng MTCNN để xử lý khuôn mặt đã cắt
                    face_tensor = mtcnn(face_img)
                    if face_tensor is not None:
                        face_tensor = face_tensor.unsqueeze(0).to(device)
                        embedding = face_model(face_tensor)
                        embedding = embedding.detach().cpu().numpy().flatten()
                        
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
    
    # Tạo từ điển lưu kết quả nhận diện cho các khuôn mặt
    face_results = {}
    
    try:
        # Chuyển ảnh từ BGR sang RGB cho MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Phát hiện khuôn mặt để lấy tọa độ bbox
        boxes, probs = mtcnn.detect(frame_rgb)
        if boxes is None or len(boxes) == 0:
            return display_frame, recognized_users
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i, box in enumerate(boxes):
            if probs[i] < 0.9:  # Lọc các khuôn mặt có xác suất thấp
                continue
                
            # Tạo key duy nhất cho mỗi khuôn mặt dựa trên vị trí
            x1, y1, x2, y2 = [int(b) for b in box]
            face_key = f"{x1}_{y1}_{x2}_{y2}"
            
            # Cắt và xử lý khuôn mặt
            face_img = frame_rgb[y1:y2, x1:x2]
            
            try:
                # Sử dụng MTCNN trực tiếp thay vì phương thức align
                face_tensor = mtcnn(face_img)
                if face_tensor is None:
                    continue
                    
                face_tensor = face_tensor.unsqueeze(0).to(device)
                embedding = face_model(face_tensor)
                face_embedding = embedding.detach().cpu().numpy().flatten()
                
                # Nhận diện khuôn mặt
                match_info, similarity = recognize_face(face_embedding, face_database)
                
                # Lưu kết quả vào từ điển
                face_results[face_key] = {
                    'box': (x1, y1, x2, y2),
                    'match_info': match_info,
                    'similarity': similarity
                }
                
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
            
            # Ensure embeddings are 1D arrays for cosine_similarity
            if isinstance(face_embedding, np.ndarray) and isinstance(db_embedding, np.ndarray):
                # Reshape if needed
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