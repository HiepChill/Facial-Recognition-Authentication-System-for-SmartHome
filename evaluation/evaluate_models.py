import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
import json
from insightface.app import FaceAnalysis
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import dlib
import torch
from tabulate import tabulate  # Để in bảng kết quả trực quan
import shutil
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_log.txt", encoding='utf-8'),
        logging.StreamHandler()
    ]
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

# Ngưỡng nhận diện - có thể điều chỉnh
RECOGNITION_THRESHOLD = 0.5

# Tạo cấu trúc thư mục mẫu
def create_sample_directory_structure():
    try:
        # Tạo thư mục dataset nếu chưa tồn tại
        os.makedirs(DATASET_DIR, exist_ok=True)
        
        # Tạo các thư mục con
        os.makedirs(KNOWN_DIR, exist_ok=True)
        os.makedirs(UNKNOWN_DIR, exist_ok=True)
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        
        # Tạo thư mục người dùng mẫu trong reference và known
        sample_users = ["user1", "user2", "user3"]
        for user in sample_users:
            os.makedirs(os.path.join(REFERENCE_DIR, user), exist_ok=True)
            os.makedirs(os.path.join(KNOWN_DIR, user), exist_ok=True)
        
        # Tạo file README.txt để hướng dẫn
        readme_content = """HƯỚNG DẪN SỬ DỤNG CÔNG CỤ ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN KHUÔN MẶT

Cấu trúc thư mục:
- dataset/
  - reference/: Chứa ảnh tham chiếu để tạo database embedding
    - user1/: Mỗi người dùng có một thư mục riêng, tên thư mục là tên người dùng
    - user2/
    ...
  - known/: Chứa ảnh kiểm tra cho người đã biết (để đánh giá true positive và false negative)
    - user1/
    - user2/
    ...
  - unknown/: Chứa ảnh người lạ (để đánh giá true negative và false positive)

Hướng dẫn:
1. Đặt ảnh tham chiếu của mỗi người trong thư mục tương ứng trong 'reference/'
2. Đặt ảnh kiểm tra (khác với ảnh tham chiếu) của mỗi người trong thư mục tương ứng trong 'known/'
3. Đặt ảnh của những người không có trong hệ thống vào thư mục 'unknown/'
4. Chạy chương trình đánh giá: python evaluate_models.py

Lưu ý:
- Đảm bảo tên thư mục trong 'known/' phải giống với tên thư mục trong 'reference/' để so sánh chính xác
- Tên thư mục là nhãn nhận diện, hãy đảm bảo đặt tên thư mục đúng với tên người dùng
- Ảnh nên ở định dạng JPG, JPEG hoặc PNG
- Mỗi người nên có ít nhất 3-5 ảnh tham chiếu để đảm bảo kết quả nhận diện tốt
"""
        
        with open(os.path.join(DATASET_DIR, "README.txt"), "w", encoding="utf-8") as f:
            f.write(readme_content)
            
        logging.info(f"Đã tạo cấu trúc thư mục mẫu tại {DATASET_DIR}")
        print(f"Đã tạo cấu trúc thư mục mẫu tại {DATASET_DIR}")
        print("Vui lòng đọc file README.txt để biết cách sử dụng.")
        
        return True
    except Exception as e:
        logging.error(f"Lỗi khi tạo cấu trúc thư mục mẫu: {str(e)}")
        return False

# Cải thiện hàm check_directories để thêm tùy chọn tạo cấu trúc mẫu
def check_directories():
    directories = [DATASET_DIR, KNOWN_DIR, UNKNOWN_DIR, REFERENCE_DIR]
    missing_dirs = [dir_path for dir_path in directories if not os.path.exists(dir_path)]
    
    if missing_dirs:
        for dir_path in missing_dirs:
            logging.warning(f"Thư mục không tồn tại: {dir_path}")
        
        # Tạo cấu trúc thư mục mẫu nếu cần
        create_sample_dirs = input("Bạn có muốn tạo cấu trúc thư mục mẫu không? (y/n): ")
        if create_sample_dirs.lower() == 'y':
            if create_sample_directory_structure():
                logging.info("Vui lòng thêm ảnh vào các thư mục trước khi chạy đánh giá.")
                return False
            else:
                logging.error("Không thể tạo cấu trúc thư mục mẫu.")
                return False
        else:
            logging.error("Đánh giá không thể tiếp tục do thiếu các thư mục dữ liệu.")
            return False
    
    # Kiểm tra xem có dữ liệu trong các thư mục không
    reference_subdirs = [d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))]
    if not reference_subdirs:
        logging.warning(f"Không tìm thấy thư mục người dùng nào trong {REFERENCE_DIR}")
        create_sample = input("Bạn có muốn tạo cấu trúc thư mục mẫu không? (y/n): ")
        if create_sample.lower() == 'y':
            if create_sample_directory_structure():
                logging.info("Vui lòng thêm ảnh vào các thư mục trước khi chạy đánh giá.")
                return False
            else:
                logging.error("Không thể tạo cấu trúc thư mục mẫu.")
                return False
        return False
        
    return True

# Lớp trừu tượng cho mô hình nhận diện
class FaceRecognitionModel:
    def initialize(self):
        pass

    def get_embedding(self, image):
        pass

    def get_name(self):
        pass

# Triển khai cho InsightFace
class InsightFaceModel(FaceRecognitionModel):
    def __init__(self):
        self.model = None

    def initialize(self):
        try:
            self.model = FaceAnalysis(name='buffalo_l', providers=[ 'DmlExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            logging.info("Khởi tạo InsightFace thành công")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo InsightFace: {str(e)}")
            raise

    def get_embedding(self, image):
        try:
            if image is None:
                return None
            faces = self.model.get(image)
            if faces:
                return faces[0].normed_embedding
            return None
        except Exception as e:
            logging.warning(f"Lỗi khi trích xuất embedding InsightFace: {str(e)}")
            return None

    def get_name(self):
        return "InsightFace"

# Triển khai cho DeepFace
class DeepFaceModel(FaceRecognitionModel):
    def __init__(self):
        self.model = None

    def initialize(self):
        try:
            self.model = "Facenet"  # Sử dụng mô hình FaceNet trong DeepFace
            logging.info("Khởi tạo DeepFace thành công")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo DeepFace: {str(e)}")
            raise

    def get_embedding(self, image):
        try:
            if image is None:
                return None
            embedding = DeepFace.represent(image, model_name=self.model, enforce_detection=False)
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]["embedding"])
            return None
        except Exception as e:
            logging.warning(f"Lỗi khi trích xuất embedding DeepFace: {str(e)}")
            return None

    def get_name(self):
        return "DeepFace"

# Triển khai cho FaceNet
class FaceNetModel(FaceRecognitionModel):
    def __init__(self):
        self.mtcnn = None
        self.model = None

    def initialize(self):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.mtcnn = MTCNN(image_size=160, margin=0, device=device)
            self.model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
            logging.info(f"Khởi tạo FaceNet thành công (sử dụng {device})")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo FaceNet: {str(e)}")
            raise

    def get_embedding(self, image):
        try:
            if image is None:
                return None
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face, prob = self.mtcnn(image_rgb, return_prob=True)
            if face is not None and prob > 0.9:
                face = face.unsqueeze(0).to(device)
                embedding = self.model(face)
                return embedding.detach().cpu().numpy().flatten()
            return None
        except Exception as e:
            logging.warning(f"Lỗi khi trích xuất embedding FaceNet: {str(e)}")
            return None

    def get_name(self):
        return "FaceNet"

# Triển khai cho Dlib
class DlibModel(FaceRecognitionModel):
    def __init__(self):
        self.face_detector = None
        self.shape_predictor = None
        self.face_recognizer = None

    def initialize(self):
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            
            shape_predictor_path = os.path.join(CURRENT_DIR, "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(shape_predictor_path):
                raise FileNotFoundError(f"File không tồn tại: {shape_predictor_path}")
            self.shape_predictor = dlib.shape_predictor(shape_predictor_path)
            
            face_recognizer_path = os.path.join(CURRENT_DIR, "dlib_face_recognition_resnet_model_v1.dat")
            if not os.path.exists(face_recognizer_path):
                raise FileNotFoundError(f"File không tồn tại: {face_recognizer_path}")
            self.face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)
            
            logging.info("Khởi tạo Dlib thành công")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo Dlib: {str(e)}")
            raise

    def get_embedding(self, image):
        try:
            if image is None:
                return None
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            if len(faces) > 0:
                shape = self.shape_predictor(image, faces[0])
                embedding = self.face_recognizer.compute_face_descriptor(image, shape)
                return np.array(embedding)
            return None
        except Exception as e:
            logging.warning(f"Lỗi khi trích xuất embedding Dlib: {str(e)}")
            return None

    def get_name(self):
        return "Dlib"

# Tải cơ sở dữ liệu embedding từ thư mục reference
def load_reference_database(model):
    face_db = {}
    user_info = {}
    
    if not os.path.exists(REFERENCE_DIR):
        logging.error(f"Thư mục tham chiếu không tồn tại: {REFERENCE_DIR}")
        return {}
    
    user_dirs = [d for d in os.listdir(REFERENCE_DIR) if os.path.isdir(os.path.join(REFERENCE_DIR, d))]
    if not user_dirs:
        logging.warning(f"Không tìm thấy thư mục người dùng nào trong: {REFERENCE_DIR}")
        return {}
    
    for user_dir in user_dirs:
        user_path = os.path.join(REFERENCE_DIR, user_dir)
        if not os.path.isdir(user_path):
            continue
            
        name = user_dir  # Chỉ lấy tên thư mục làm name
        logging.info(f"Đang tải embedding cho người dùng: {name}")
        
        img_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files:
            logging.warning(f"Không tìm thấy ảnh trong thư mục: {user_path}")
            continue
            
        valid_embeddings = 0
        for img_name in img_files:
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Không thể đọc ảnh: {img_path}")
                continue
                
            embedding = model.get_embedding(img)
            if embedding is not None:
                if name not in face_db:
                    face_db[name] = []
                    user_info[name] = {"name": name}
                face_db[name].append(embedding)
                valid_embeddings += 1
        
        if valid_embeddings > 0:
            logging.info(f"Đã tạo {valid_embeddings} embedding cho người dùng {name}")
        else:
            logging.warning(f"Không tạo được embedding nào cho người dùng {name}")
    
    # Tính trung bình các embedding
    averaged_db = {}
    for name, embeddings in face_db.items():
        if embeddings:
            averaged_embedding = np.mean(embeddings, axis=0)
            averaged_db[name] = {
                "embedding": averaged_embedding,
                "info": user_info[name]
            }
    
    logging.info(f"Đã tạo cơ sở dữ liệu embedding cho {len(averaged_db)} người dùng")
    return averaged_db

# Nhận diện khuôn mặt
def recognize_face(embedding, face_database, threshold=RECOGNITION_THRESHOLD):
    if embedding is None:
        return None, -1
        
    if not face_database:
        logging.warning("Cơ sở dữ liệu embedding trống")
        return None, -1
        
    match_info = None
    max_similarity = -1
    
    for name, data in face_database.items():
        db_embedding = data["embedding"]
        
        # Tính toán độ tương đồng cosine
        similarity = cosine_similarity([embedding], [db_embedding])[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > threshold:
                match_info = data["info"]
            else:
                match_info = None
    
    return match_info, max_similarity

# Đánh giá mô hình
def evaluate_model(model, threshold=RECOGNITION_THRESHOLD):
    results = {
        "model_name": model.get_name(),
        "init_time": 0.0,
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
        "avg_processing_time": 0.0,
        "threshold": threshold
    }

    try:
        # Đo thời gian khởi tạo mô hình
        logging.info(f"Khởi tạo mô hình {model.get_name()}")
        start_time = time.time()
        model.initialize()
        init_time = time.time() - start_time
        results["init_time"] = init_time
        logging.info(f"Khởi tạo mô hình {model.get_name()} hoàn tất trong {init_time:.2f} giây")

        # Tạo cơ sở dữ liệu embedding
        logging.info(f"Đang tạo cơ sở dữ liệu embedding cho {model.get_name()}")
        face_database = load_reference_database(model)
        if not face_database:
            logging.error(f"Không thể tạo cơ sở dữ liệu embedding cho {model.get_name()}")
            return results

        # Tải ảnh đánh giá
        known_images = []
        unknown_images = []
        
        # Kiểm tra và tải ảnh người đã biết
        if os.path.exists(KNOWN_DIR):
            for user_dir in os.listdir(KNOWN_DIR):
                user_path = os.path.join(KNOWN_DIR, user_dir)
                if os.path.isdir(user_path):
                    img_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if not img_files:
                        logging.warning(f"Không tìm thấy ảnh trong thư mục: {user_path}")
                        continue
                    for img_name in img_files:
                        known_images.append((os.path.join(user_path, img_name), user_dir))
        else:
            logging.error(f"Thư mục {KNOWN_DIR} không tồn tại")
            
        # Kiểm tra và tải ảnh người lạ
        if os.path.exists(UNKNOWN_DIR):
            img_files = [f for f in os.listdir(UNKNOWN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files:
                logging.warning(f"Không tìm thấy ảnh trong thư mục: {UNKNOWN_DIR}")
            else:
                for img_name in img_files:
                    unknown_images.append((os.path.join(UNKNOWN_DIR, img_name), None))
        else:
            logging.error(f"Thư mục {UNKNOWN_DIR} không tồn tại")
            
        logging.info(f"Đã tải {len(known_images)} ảnh người đã biết và {len(unknown_images)} ảnh người lạ")
        
        if len(known_images) == 0 and len(unknown_images) == 0:
            logging.error("Không có ảnh để đánh giá")
            return results

        # Đánh giá trên ảnh của người đã biết
        if known_images:
            logging.info(f"Đánh giá {model.get_name()} trên ảnh người đã biết")
            for img_path, true_label in tqdm(known_images, desc=f"Đánh giá {model.get_name()} trên ảnh known"):
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"Không thể đọc ảnh: {img_path}")
                    continue

                start_time = time.time()
                embedding = model.get_embedding(image)
                match_info, similarity = recognize_face(embedding, face_database, threshold)
                end_time = time.time()

                results["total_images"] += 1
                results["processing_time"] += (end_time - start_time)

                true_name = true_label if true_label else None
                pred_name = match_info["name"] if match_info else None

                if pred_name and pred_name == true_name:
                    results["true_positives"] += 1
                else:
                    results["false_negatives"] += 1
                    logging.debug(f"False Negative: {img_path} - Thực tế: {true_name}, Nhận diện: {pred_name}")

        # Đánh giá trên ảnh của người lạ
        if unknown_images:
            logging.info(f"Đánh giá {model.get_name()} trên ảnh người lạ")
            for img_path, true_label in tqdm(unknown_images, desc=f"Đánh giá {model.get_name()} trên ảnh unknown"):
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"Không thể đọc ảnh: {img_path}")
                    continue

                start_time = time.time()
                embedding = model.get_embedding(image)
                match_info, similarity = recognize_face(embedding, face_database, threshold)
                end_time = time.time()

                results["total_images"] += 1
                results["processing_time"] += (end_time - start_time)

                if match_info:
                    results["false_positives"] += 1
                    logging.debug(f"False Positive: {img_path} - Nhận diện nhầm là: {match_info['name']}")
                else:
                    results["true_negatives"] += 1

        # Tính toán các chỉ số
        total_predictions = results["total_images"]
        if total_predictions > 0:
            results["accuracy"] = (results["true_positives"] + results["true_negatives"]) / total_predictions
            results["precision"] = results["true_positives"] / (results["true_positives"] + results["false_positives"]) if (results["true_positives"] + results["false_positives"]) > 0 else 0
            results["recall"] = results["true_positives"] / (results["true_positives"] + results["false_negatives"]) if (results["true_positives"] + results["false_negatives"]) > 0 else 0
            results["f1_score"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"]) if (results["precision"] + results["recall"]) > 0 else 0
            results["avg_processing_time"] = results["processing_time"] / total_predictions
            
        logging.info(f"Đánh giá {model.get_name()} hoàn tất")
        
    except Exception as e:
        logging.error(f"Lỗi khi đánh giá mô hình {model.get_name()}: {str(e)}")
        
    return results

# Hàm chính
def main():
    print("====== CHƯƠNG TRÌNH ĐÁNH GIÁ CÁC MÔ HÌNH NHẬN DIỆN KHUÔN MẶT ======")
    
    # Hiển thị menu chính
    print("\nLựa chọn thao tác:")
    print("1. Đánh giá các mô hình")
    print("2. Tạo cấu trúc thư mục mẫu")
    print("3. Hiển thị hướng dẫn")
    print("4. Thoát")
    
    choice = input("Nhập lựa chọn của bạn (1-4): ")
    
    if choice == "1":
        # Kiểm tra các thư mục dữ liệu trước khi đánh giá
        if not check_directories():
            return
        
        # Danh sách các mô hình
        models = [
            InsightFaceModel(),
            DeepFaceModel(),
            FaceNetModel(),
            DlibModel()
        ]
        
        # Hỏi người dùng muốn đánh giá những mô hình nào
        print("\nCác mô hình có sẵn để đánh giá:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model.get_name()}")
        
        selected_models = input("Chọn các mô hình để đánh giá (nhập số, cách nhau bằng dấu phẩy, hoặc 'all' cho tất cả): ")
        
        if selected_models.lower() == 'all':
            selected_indices = list(range(len(models)))
        else:
            try:
                selected_indices = [int(idx.strip()) - 1 for idx in selected_models.split(',')]
                selected_indices = [idx for idx in selected_indices if 0 <= idx < len(models)]
            except:
                logging.error("Lựa chọn không hợp lệ, sẽ đánh giá tất cả các mô hình")
                selected_indices = list(range(len(models)))
        
        if not selected_indices:
            logging.error("Không có mô hình nào được chọn")
            return
        
        # Hỏi về ngưỡng nhận diện
        threshold_input = input(f"Nhập ngưỡng nhận diện (mặc định: {RECOGNITION_THRESHOLD}): ")
        try:
            threshold = float(threshold_input) if threshold_input else RECOGNITION_THRESHOLD
            if not (0 <= threshold <= 1):
                logging.warning(f"Ngưỡng không hợp lệ. Sử dụng ngưỡng mặc định: {RECOGNITION_THRESHOLD}")
                threshold = RECOGNITION_THRESHOLD
        except:
            logging.warning(f"Ngưỡng không hợp lệ. Sử dụng ngưỡng mặc định: {RECOGNITION_THRESHOLD}")
            threshold = RECOGNITION_THRESHOLD
        
        # Đánh giá các mô hình đã chọn
        evaluation_results = []
        for idx in selected_indices:
            model = models[idx]
            print(f"\nĐang đánh giá {model.get_name()}...")
            result = evaluate_model(model, threshold)
            evaluation_results.append(result)

        # In kết quả dưới dạng bảng
        if evaluation_results:
            table_data = []
            for result in evaluation_results:
                table_data.append([
                    result["model_name"],
                    f"{result['accuracy']:.4f}",
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['f1_score']:.4f}",
                    result["false_positives"],
                    result["false_negatives"],
                    f"{result['init_time']:.2f}",
                    f"{result['avg_processing_time']:.4f}"
                ])

            headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "False Positives", "False Negatives", "Init Time (s)", "Avg Proc Time (s)"]
            print("\nKết Quả Đánh Giá:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Lưu kết quả
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
            with open(result_filename, "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, indent=4)
            print(f"Kết quả đã được lưu vào {result_filename}")
        else:
            print("Không có kết quả đánh giá")
    
    elif choice == "2":
        create_sample_directory_structure()
    
    elif choice == "3":
        print("""
HƯỚNG DẪN SỬ DỤNG CÔNG CỤ ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN KHUÔN MẶT

Cấu trúc thư mục:
- dataset/
  - reference/: Chứa ảnh tham chiếu để tạo database embedding
    - user1/: Mỗi người dùng có một thư mục riêng, tên thư mục là tên người dùng
    - user2/
    ...
  - known/: Chứa ảnh kiểm tra cho người đã biết (để đánh giá true positive và false negative)
    - user1/
    - user2/
    ...
  - unknown/: Chứa ảnh người lạ (để đánh giá true negative và false positive)

Hướng dẫn:
1. Đặt ảnh tham chiếu của mỗi người trong thư mục tương ứng trong 'reference/'
2. Đặt ảnh kiểm tra (khác với ảnh tham chiếu) của mỗi người trong thư mục tương ứng trong 'known/'
3. Đặt ảnh của những người không có trong hệ thống vào thư mục 'unknown/'
4. Chạy chương trình đánh giá: python evaluate_models.py

Lưu ý:
- Đảm bảo tên thư mục trong 'known/' phải giống với tên thư mục trong 'reference/' để so sánh chính xác
- Tên thư mục là nhãn nhận diện, hãy đảm bảo đặt tên thư mục đúng với tên người dùng
- Ảnh nên ở định dạng JPG, JPEG hoặc PNG
- Mỗi người nên có ít nhất 3-5 ảnh tham chiếu để đảm bảo kết quả nhận diện tốt
        """)
        input("Nhấn Enter để tiếp tục...")
        main()  # Quay lại menu chính
    
    elif choice == "4":
        print("Kết thúc chương trình.")
    
    else:
        print("Lựa chọn không hợp lệ!")
        main()  # Quay lại menu chính

if __name__ == "__main__":
    main()