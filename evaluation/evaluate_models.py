import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
import json
from insightface.app import FaceAnalysis
from tabulate import tabulate
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Lấy đường dẫn tương đối
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CURRENT_DIR, "dataset")
KNOWN_DIR = os.path.join(DATASET_DIR, "known")
UNKNOWN_DIR = os.path.join(DATASET_DIR, "unknown")
REFERENCE_DIR = os.path.join(DATASET_DIR, "reference")
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")

# Tạo thư mục kết quả nếu chưa tồn tại
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ngưỡng nhận diện
RECOGNITION_THRESHOLD = 0.6

# Tạo cấu trúc thư mục mẫu nếu chưa có
def create_sample_directory_structure():
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(KNOWN_DIR, exist_ok=True)
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        
        # Tạo thư mục người dùng mẫu
        sample_users = ["user1", "user2", "user3"]
        for user in sample_users:
            os.makedirs(os.path.join(REFERENCE_DIR, user), exist_ok=True)
            os.makedirs(os.path.join(KNOWN_DIR, user), exist_ok=True)
        
        logging.info(f"Đã tạo cấu trúc thư mục mẫu tại {DATASET_DIR}")
        return True
    except Exception as e:
        logging.error(f"Lỗi khi tạo cấu trúc thư mục: {str(e)}")
        return False

# Kiểm tra thư mục
def check_directories():
    directories = [DATASET_DIR, KNOWN_DIR, UNKNOWN_DIR, REFERENCE_DIR]
    missing_dirs = [dir_path for dir_path in directories if not os.path.exists(dir_path)]
    
    if missing_dirs:
        logging.info("Tạo cấu trúc thư mục mẫu...")
        create_sample_directory_structure()
        return False
    
    # Kiểm tra có dữ liệu không
    reference_subdirs = [d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))]
    if not reference_subdirs:
        logging.info("Tạo cấu trúc thư mục mẫu...")
        create_sample_directory_structure()
        return False
        
    return True

# Khởi tạo InsightFace
def initialize_insightface():
    try:
        model = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider'])
        model.prepare(ctx_id=0, det_size=(640, 640))
        logging.info("Khởi tạo InsightFace thành công")
        return model
    except Exception as e:
        logging.error(f"Lỗi khởi tạo InsightFace: {str(e)}")
        return None

# Trích xuất embedding
def get_embedding(model, image):
    try:
        if image is None:
            return None
        faces = model.get(image)
        if faces:
            return faces[0].normed_embedding
        return None
    except Exception as e:
        logging.warning(f"Lỗi khi trích xuất embedding: {str(e)}")
        return None

# Tải cơ sở dữ liệu embedding từ thư mục reference
def load_reference_database(model):
    face_db = {}
    
    user_dirs = [d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))]
    if not user_dirs:
        logging.warning(f"Không tìm thấy thư mục người dùng nào trong: {REFERENCE_DIR}")
        return {}
    
    for user_dir in user_dirs:
        user_path = os.path.join(REFERENCE_DIR, user_dir)
        name = user_dir
        logging.info(f"Đang tải embedding cho người dùng: {name}")
        
        img_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            logging.warning(f"Không tìm thấy ảnh trong thư mục: {user_path}")
            continue
            
        embeddings = []
        for img_name in img_files:
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            embedding = get_embedding(model, img)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            # Tính trung bình các embedding
            averaged_embedding = np.mean(embeddings, axis=0)
            face_db[name] = {
                "embedding": averaged_embedding,
                "name": name
            }
            logging.info(f"Đã tạo {len(embeddings)} embedding cho người dùng {name}")
    
    logging.info(f"Đã tạo cơ sở dữ liệu embedding cho {len(face_db)} người dùng")
    return face_db

# Nhận diện khuôn mặt
def recognize_face(embedding, face_database, threshold=RECOGNITION_THRESHOLD):
    if embedding is None or not face_database:
        return None, -1
        
    max_similarity = -1
    match_info = None
    
    for name, data in face_database.items():
        db_embedding = data["embedding"]
        similarity = cosine_similarity([embedding], [db_embedding])[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > threshold:
                match_info = data
            else:
                match_info = None
    
    return match_info, max_similarity

# Đánh giá mô hình InsightFace
def evaluate_insightface():
    results = {
        "model_name": "InsightFace",
        "total_images": 0,
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "processing_time": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "threshold": RECOGNITION_THRESHOLD
    }

    # Khởi tạo mô hình
    model = initialize_insightface()
    if not model:
        logging.error("Không thể khởi tạo InsightFace")
        return results

    # Tạo cơ sở dữ liệu embedding
    face_database = load_reference_database(model)
    if not face_database:
        logging.error("Không thể tạo cơ sở dữ liệu embedding")
        return results

    # Tải ảnh đánh giá
    known_images = []
    unknown_images = []
    
    # Ảnh người đã biết
    if os.path.exists(KNOWN_DIR):
        for user_dir in os.listdir(KNOWN_DIR):
            user_path = os.path.join(KNOWN_DIR, user_dir)
            if os.path.isdir(user_path):
                img_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_name in img_files:
                    known_images.append((os.path.join(user_path, img_name), user_dir))
    
    # Ảnh người lạ
    if os.path.exists(UNKNOWN_DIR):
        img_files = [f for f in os.listdir(UNKNOWN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in img_files:
            unknown_images.append((os.path.join(UNKNOWN_DIR, img_name), None))
            
    logging.info(f"Đã tải {len(known_images)} ảnh người đã biết và {len(unknown_images)} ảnh người lạ")
    
    # Đánh giá trên ảnh người đã biết
    for img_path, true_label in tqdm(known_images, desc="Đánh giá ảnh known"):
        image = cv2.imread(img_path)
        if image is None:
            continue

        start_time = time.time()
        embedding = get_embedding(model, image)
        match_info, similarity = recognize_face(embedding, face_database)
        results["processing_time"] += (time.time() - start_time)

        results["total_images"] += 1
        pred_name = match_info["name"] if match_info else None

        if pred_name and pred_name == true_label:
            results["true_positives"] += 1
        else:
            results["false_negatives"] += 1

    # Đánh giá trên ảnh người lạ
    for img_path, _ in tqdm(unknown_images, desc="Đánh giá ảnh unknown"):
        image = cv2.imread(img_path)
        if image is None:
            continue

        start_time = time.time()
        embedding = get_embedding(model, image)
        match_info, similarity = recognize_face(embedding, face_database)
        results["processing_time"] += (time.time() - start_time)

        results["total_images"] += 1

        if match_info:
            results["false_positives"] += 1
        else:
            results["true_negatives"] += 1

    # Tính toán các chỉ số
    total = results["total_images"]
    if total > 0:
        results["accuracy"] = (results["true_positives"] + results["true_negatives"]) / total
        
        tp_fp = results["true_positives"] + results["false_positives"]
        results["precision"] = results["true_positives"] / tp_fp if tp_fp > 0 else 0
        
        tp_fn = results["true_positives"] + results["false_negatives"]
        results["recall"] = results["true_positives"] / tp_fn if tp_fn > 0 else 0
        
        pr_re = results["precision"] + results["recall"]
        results["f1_score"] = 2 * (results["precision"] * results["recall"]) / pr_re if pr_re > 0 else 0
        
    return results

# Hàm chính
def main():
    print("====== ĐÁNH GIÁ MÔ HÌNH INSIGHTFACE ======")
    
    # Kiểm tra thư mục
    if not check_directories():
        print("Vui lòng thêm ảnh vào các thư mục dataset/reference/, dataset/known/, dataset/unknown/ trước khi chạy đánh giá.")
        return

    # Đánh giá InsightFace
    print("Đang đánh giá InsightFace...")
    result = evaluate_insightface()

    # In kết quả
    if result["total_images"] > 0:
        table_data = [[
            result["model_name"],
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1_score']:.4f}",
            result["true_positives"],
            result["true_negatives"],
            result["false_positives"],
            result["false_negatives"],
            f"{result['processing_time']:.2f}s"
        ]]

        headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "TP", "TN", "FP", "FN", "Total Time"]
        print("\nKết Quả Đánh Giá:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Lưu kết quả
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = os.path.join(RESULTS_DIR, f"insightface_evaluation_{timestamp}.json")
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"\nKết quả đã được lưu vào {result_filename}")
    else:
        print("Không có dữ liệu để đánh giá!")

if __name__ == "__main__":
    main()