import sqlite3
import os
import shutil
import csv
from .config import DB_PATH, DATASET_DIR
from datetime import datetime


def setup_database():
    """Thiết lập cơ sở dữ liệu"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        image_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS detection_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        name TEXT,
        is_known BOOLEAN,
        confidence REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT,
        notification_sent BOOLEAN DEFAULT 0
    )
    ''')
    cursor.execute('''
    CREATE TRIGGER IF NOT EXISTS update_user_timestamp
    AFTER UPDATE ON users
    BEGIN
        UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
    END;
    ''')
    conn.commit()
    conn.close()

def check_user_exists(user_id):
    """Kiểm tra xem user_id đã tồn tại chưa"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def add_user(user_id, name, notes=None):
    """Thêm người dùng mới"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (id, name, notes) VALUES (?, ?, ?)", (user_id, name, notes))
    conn.commit()
    conn.close()

def update_user(user_id, name=None, notes=None):
    """Cập nhật thông tin người dùng"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, notes FROM users WHERE id = ?", (user_id,))
    current_data = cursor.fetchone()
    if not current_data:
        conn.close()
        return False, "Không tìm thấy người dùng"
    current_name, current_notes = current_data
    new_name = name if name is not None else current_name
    new_notes = notes if notes is not None else current_notes
    cursor.execute("UPDATE users SET name = ?, notes = ? WHERE id = ?", (new_name, new_notes, user_id))
    conn.commit()
    conn.close()
    return True, "Cập nhật người dùng thành công"

def delete_user(user_id):
    """Xóa người dùng và tất cả ảnh khuôn mặt liên quan"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        if not user_data:
            conn.close()
            return False, "Không tìm thấy người dùng"
        user_name = user_data[0]
        cursor.execute("SELECT image_path FROM face_images WHERE user_id = ?", (user_id,))
        face_images = cursor.fetchall()
        cursor.execute("DELETE FROM face_images WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        user_folder = os.path.join(DATASET_DIR, f"{user_id}_{user_name}")
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
        conn.commit()
        conn.close()
        return True, f"Xóa người dùng {user_id} thành công với {len(face_images)} ảnh khuôn mặt"
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Lỗi khi xóa người dùng: {str(e)}"

def add_face_image(user_id, image_path):
    """Thêm đường dẫn ảnh khuôn mặt vào cơ sở dữ liệu"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO face_images (user_id, image_path) VALUES (?, ?)", (user_id, image_path))
    conn.commit()
    conn.close()
    return True

def delete_face_image(image_id):
    """Xóa một ảnh khuôn mặt cụ thể"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT image_path FROM face_images WHERE id = ?", (image_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False, "Không tìm thấy ảnh"
        image_path = result[0]
        cursor.execute("DELETE FROM face_images WHERE id = ?", (image_id,))
        if os.path.exists(image_path):
            os.remove(image_path)
        conn.commit()
        conn.close()
        return True, "Xóa ảnh thành công"
    except Exception as e:
        conn.rollback()
        conn.close()
        return False, f"Lỗi khi xóa ảnh: {str(e)}"

def get_all_users():
    """Lấy danh sách tất cả người dùng với thông tin chi tiết"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
    SELECT u.id, u.name, u.notes, u.created_at, u.updated_at,
    COUNT(f.id) as image_count
    FROM users u
    LEFT JOIN face_images f ON u.id = f.user_id
    GROUP BY u.id
    ORDER BY u.name
    """)
    users = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return users