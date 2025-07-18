import requests
import time
import os
from datetime import datetime
import threading
import logging
import queue
from .config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NOTIFICATION_INTERVAL_KNOWN, NOTIFICATION_INTERVAL_UNKNOWN, SECURITY_LOG_DIR

# Queue để xử lý thông báo tuần tự
notification_queue = queue.Queue()
notification_worker_running = False
last_notification_time = {}
notification_lock = threading.Lock()

def start_notification_worker():
    """Khởi động worker thread để xử lý queue thông báo"""
    global notification_worker_running
    if not notification_worker_running:
        notification_worker_running = True
        worker_thread = threading.Thread(target=notification_worker, daemon=True)
        worker_thread.start()

def notification_worker():
    """Worker thread xử lý thông báo từ queue"""
    while notification_worker_running:
        try:
            # Lấy thông báo từ queue (chờ tối đa 1 giây)
            message, image_path = notification_queue.get(timeout=1)
            
            # Gửi thông báo
            send_telegram_notification_sync(message, image_path)
            
            # Rate limiting - chờ 1 giây giữa các thông báo
            time.sleep(1)
            
            notification_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Lỗi trong notification worker: {e}")

def send_telegram_notification(message, image_path=None):
    """Thêm thông báo vào queue"""
    start_notification_worker()
    notification_queue.put((message, image_path))

def send_telegram_notification_sync(message, image_path=None):
    """Gửi thông báo đồng bộ với retry mechanism"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                logging.error("Token bot Telegram hoặc ID chat chưa được cấu hình")
                return False
            
            if image_path and os.path.exists(image_path):
                # Gửi ảnh với caption thay vì gửi riêng biệt
                return send_photo_with_caption(message, image_path)
            else:
                # Chỉ gửi tin nhắn văn bản
                return send_text_message(message)
                
        except Exception as e:
            logging.error(f"Thử lần {attempt + 1} thất bại: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return False
    
    return False

def send_photo_with_caption(message, image_path):
    """Gửi ảnh kèm caption"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        with open(image_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": message,
                "parse_mode": "HTML"
            }
            
            response = requests.post(url, data=data, files=files, timeout=30)
            
            if response.status_code == 200:
                logging.info(f"Đã gửi ảnh với caption: {message}")
                return True
            else:
                logging.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        logging.error(f"Lỗi gửi ảnh: {e}")
        return False

def send_text_message(message):
    """Gửi chỉ tin nhắn văn bản"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=10)
        
        if response.status_code == 200:
            logging.info(f"Đã gửi tin nhắn: {message}")
            return True
        else:
            logging.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logging.error(f"Lỗi gửi tin nhắn: {e}")
        return False

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