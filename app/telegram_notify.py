import requests
import time
import os
from datetime import datetime
import threading
import logging
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NOTIFICATION_INTERVAL_KNOWN, NOTIFICATION_INTERVAL_UNKNOWN, SECURITY_LOG_DIR

# Từ điển lưu thời gian thông báo cuối cùng cho mỗi người
last_notification_time = {}
notification_lock = threading.Lock()

def setup_security_logs():
    """Tạo thư mục nhật ký bảo mật nếu chưa tồn tại"""
    os.makedirs(SECURITY_LOG_DIR, exist_ok=True)
    
    # Thiết lập logging
    log_file = os.path.join(SECURITY_LOG_DIR, f"security_{datetime.now().strftime('%Y-%m-%d')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def send_telegram_notification(message, image_path=None):
    """Gửi thông báo và ảnh (nếu có) qua Telegram"""
    try:
        # Kiểm tra cấu hình cần thiết
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            logging.error("Token bot Telegram hoặc ID chat chưa được cấu hình")
            return False
            
        # Gửi tin nhắn văn bản
        text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        text_payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(text_url, data=text_payload)
        
        # Nếu có đường dẫn ảnh, gửi ảnh
        if image_path and os.path.exists(image_path):
            image_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                image_payload = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": "Ảnh đã chụp"
                }
                image_response = requests.post(image_url, data=image_payload, files=files)
                
        logging.info(f"Đã gửi thông báo Telegram: {message}")
        return True
    except Exception as e:
        logging.error(f"Gửi thông báo Telegram thất bại: {str(e)}")
        return False

def can_send_notification(person_id, is_known=True):
    """Kiểm tra xem có thể gửi thông báo dựa trên khoảng thời gian không"""
    current_time = time.time()
    
    with notification_lock:
        if person_id in last_notification_time:
            last_time = last_notification_time[person_id]
            time_diff = current_time - last_time
            
            # Khoảng thời gian khác nhau cho người quen và người lạ
            required_interval = NOTIFICATION_INTERVAL_KNOWN if is_known else NOTIFICATION_INTERVAL_UNKNOWN
            
            if time_diff < required_interval:
                return False
        
        # Cập nhật thời gian thông báo cuối cùng
        last_notification_time[person_id] = current_time
        return True

def notify_recognized_person(name, user_id, confidence, image_path=None):
    """Thông báo về người được nhận diện"""
    if can_send_notification(user_id, is_known=True):
        message = f"✅ <b>Người Được Nhận Diện</b>\nTên: {name}\nID: {user_id}\nĐộ Tin Cậy: {confidence:.2f}\nThời Gian: {datetime.now().strftime('%H:%M:%S')}"
        return send_telegram_notification(message, image_path)
    return False

def notify_unknown_person(image_path=None):
    """Thông báo về người lạ"""
    # Với người lạ, dùng timestamp làm ID để giới hạn thông báo
    unknown_id = "unknown_person"
    
    if can_send_notification(unknown_id, is_known=False):
        message = f"⚠️ <b>CẢNH BÁO: Phát Hiện Người Lạ!</b>\nThời Gian: {datetime.now().strftime('%H:%M:%S')}"
        return send_telegram_notification(message, image_path)
    return False

def log_security_event(event_type, details):
    """Ghi lại sự kiện bảo mật vào file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"[{event_type}] {details}")