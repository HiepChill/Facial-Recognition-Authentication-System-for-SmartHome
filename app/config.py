import os


# Cấu hình đường dẫn
DB_PATH = "database.db"
DATASET_DIR = "./dataset"


# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATASET_DIR, exist_ok=True)
