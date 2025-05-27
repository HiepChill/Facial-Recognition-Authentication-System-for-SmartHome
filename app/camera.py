import cv2
import threading
import time
import numpy as np

class CameraManager:
    _instance = None
    _camera = None
    _lock = threading.Lock()
    _last_frame = None
    _last_frame_time = 0
    _frame_buffer_size = 3
    _frame_buffer = []
    
    @classmethod
    def get_instance(cls):
        """Lấy instance của CameraManager (mẫu Singleton)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Khởi tạo CameraManager"""
        if CameraManager._instance is not None:
            raise Exception("Lớp Singleton - sử dụng get_instance()")
        self._camera = None
        # Thiết lập thread để đọc frame từ camera
        self._running = False
        self._thread = None
    
    def _camera_reader(self):
        """Thread để đọc frame từ camera liên tục"""
        while self._running:
            if self._camera is not None and self._camera.isOpened():
                success, frame = self._camera.read()
                if success:
                    with self._lock:
                        self._last_frame = frame
                        self._last_frame_time = time.time()
                        
                        # Thêm vào buffer và giữ kích thước cố định
                        self._frame_buffer.append(frame)
                        if len(self._frame_buffer) > self._frame_buffer_size:
                            self._frame_buffer.pop(0)
            time.sleep(0.01)  # Ngăn thread tiêu thụ 100% CPU
    
    def start_camera_thread(self):
        """Bắt đầu thread đọc frame từ camera"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._camera_reader, daemon=True)
            self._thread.start()
    
    def get_camera(self):
        """Lấy camera, khởi tạo nếu chưa có"""
        if self._camera is None:
            with self._lock:
                if self._camera is None:
                    self._camera = cv2.VideoCapture(0)
                    # Thiết lập các tham số camera
                    self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self._camera.set(cv2.CAP_PROP_FPS, 30)
                    self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                    
                    # Bắt đầu thread đọc frame
                    self.start_camera_thread()
        
        if not self._camera.isOpened():
            raise Exception("Không thể mở camera")
        return self._camera
    
    def get_frame(self):
        """Lấy frame hiện tại từ buffer thay vì đọc trực tiếp từ camera"""
        with self._lock:
            if self._last_frame is not None:
                return True, self._last_frame.copy()
            if len(self._frame_buffer) > 0:
                return True, self._frame_buffer[-1].copy()
        return False, None
    
    def release_camera(self):
        """Giải phóng tài nguyên camera"""
        with self._lock:
            if self._camera is not None:
                self._running = False
                if self._thread and self._thread.is_alive():
                    self._thread.join(timeout=1.0)
                self._camera.release()
                self._camera = None
                self._last_frame = None
                self._frame_buffer.clear()
                print("Camera đã được giải phóng")