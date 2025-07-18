import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Path, Query, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from contextlib import asynccontextmanager
from datetime import datetime
import os
import numpy as np
import shutil
import uuid
from app.config import DATASET_DIR, TEMP_DIR
from app.database import (
    setup_database, check_user_exists, add_user, add_face_image, 
    get_all_users, get_user_face_images, get_user_face_data, get_user_details,
    update_user, delete_user, delete_face_image, get_detection_history
)
from app.face_recognition import face_app, face_database, load_face_database, process_frame
from app.camera import CameraManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_database
    setup_database()
    user_face_data = get_user_face_data()
    face_database = load_face_database(user_face_data)
    print(f"Đã tải {len(face_database)} người dùng vào CSDL")
    yield
    camera_manager = CameraManager.get_instance()
    camera_manager.release_camera()
    print("Ứng dụng đang tắt: Đã giải phóng tài nguyên camera")

app = FastAPI(
    title="Smart Home Security System", 
    lifespan=lifespan, 
    description="Hệ thống an ninh nhà thông minh sử dụng nhận diện khuôn mặt và thông báo Telegram."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Cho phép tất cả domain truy cập
    allow_credentials=True,     # Cho phép gửi cookie / header Authorization
    allow_methods=["*"],        # Cho phép tất cả phương thức HTTP (GET, POST, PUT, DELETE, ...)
    allow_headers=["*"],        # Cho phép tất cả loại header từ client gửi lên
)

app.mount("/images", StaticFiles(directory=DATASET_DIR), name="images")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

#Thêm người mới
@app.post("/users")
async def create_user(
    user_id: str = Form(...),
    name: str = Form(...),
    notes: str = Form(None)
):
    if check_user_exists(user_id):
        raise HTTPException(status_code=400, detail="User ID đã tồn tại")
    add_user(user_id, name, notes)
    user_folder = os.path.join(DATASET_DIR, f"{user_id}_{name}")
    os.makedirs(user_folder, exist_ok=True)
    return {"status": "success", "message": "Người dùng đã được đăng ký", "user_id": user_id}

#Lấy danh sách người dùng
@app.get("/users")
async def list_users():
    users = get_all_users()
    return {"status": "success", "users": users, "total": len(users)}

#Lấy thông tin người dùng
@app.get("/users/{user_id}")
async def get_user(user_id: str = Path(...)):
    user = get_user_details(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    return {"status": "success", "user": user}

#Cập nhật thông tin người dùng
@app.put("/users/{user_id}")
async def update_user_info(
    user_id: str = Path(...),
    name: str = Form(None),
    notes: str = Form(None)
):
    if not check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    success, message = update_user(user_id, name, notes)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "success", "message": message}

#Xóa người dùng
@app.delete("/users/{user_id}")
async def remove_user(user_id: str = Path(...)):
    if not check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    success, message = delete_user(user_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    global face_database
    user_face_data = get_user_face_data()
    face_database = load_face_database(user_face_data)
    return {"status": "success", "message": message}

#Thêm ảnh khuôn mặt
@app.post("/users/{user_id}/faces")
async def add_face(
    user_id: str = Path(...),
    face_images: List[UploadFile] = File(...)  # Nhận danh sách ảnh
):
    if not check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    
    user = get_user_details(user_id)
    user_folder = os.path.join(DATASET_DIR, f"{user_id}_{user['name']}")
    os.makedirs(user_folder, exist_ok=True)
    
    results = []
    for face_image in face_images:
        try:
            content = await face_image.read()
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            
            # Sử dụng InsightFace để phát hiện khuôn mặt (chỉ để validation)
            faces = face_app.get(img)
            if not faces or len(faces) == 0:
                results.append({"filename": face_image.filename, "status": "error", "message": "Không tìm thấy khuôn mặt"})
                continue

            # Tạo filename với format ID_ten_001
            user_name = user['name']
            
            # Đếm số ảnh hiện có của user để tạo số thứ tự
            existing_images = get_user_face_images(user_id)
            next_number = len(existing_images) + 1
            
            filename = f"{user_id}_{user_name}_{next_number:03d}.jpg"
            image_path = os.path.join(user_folder, filename)
            cv2.imwrite(image_path, img) 
            add_face_image(user_id, image_path)
            results.append({"filename": face_image.filename, "status": "success", "image_path": image_path})
        except Exception as e:
            results.append({"filename": face_image.filename, "status": "error", "message": str(e)})
    
    global face_database
    user_face_data = get_user_face_data()
    face_database = load_face_database(user_face_data)
    
    return {"status": "success", "results": results}

#Lấy danh sách ảnh khuôn mặt
@app.get("/users/{user_id}/faces")
async def list_faces(user_id: str = Path(...)):
    if not check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    face_images = get_user_face_images(user_id)
    return {"status": "success", "face_images": face_images, "total": len(face_images)}

#Xóa ảnh khuôn mặt
@app.delete("/faces/{image_id}")
async def remove_face(image_id: int = Path(...)):
    success, message = delete_face_image(image_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    global face_database
    user_face_data = get_user_face_data()
    face_database = load_face_database(user_face_data)
    return {"status": "success", "message": message}

def generate_frames():
    camera_manager = CameraManager.get_instance()
    camera = camera_manager.get_camera()
    while True:
        success, frame = camera_manager.get_frame()
        if not success:
            continue
        global face_database
        processed_frame, recognized_users = process_frame(frame, face_database)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#Lấy stream video
@app.get("/camera/stream")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

#Lấy lịch sử nhận diện
@app.get("/recognition/history")
async def get_recognition_history(date: str = Query(None, description="Ngày cần truy vấn (YYYY-MM-DD)")):
    try:
        # Nếu không cung cấp ngày, sử dụng ngày hiện tại
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
            
        history = get_detection_history(date=date)
        
        # Kiểm tra xem history có phải là danh sách không
        if not isinstance(history, list):
            history = []
            
        return {"status": "success", "history": history, "total": len(history), "date": date}
    except Exception as e:
        # Sử dụng ngày hiện tại nếu có lỗi với date
        current_date = datetime.now().strftime("%Y-%m-%d")
        return {
            "status": "error", 
            "message": f"Lỗi khi truy vấn lịch sử nhận diện: {str(e)}",
            "history": [],
            "total": 0,
            "date": date if date else current_date
        }

@app.get("/")
async def root():
    return {
        "name": "Smart Home Security System",
        "version": "1.0.0",
        "description": "Smart Home Security System by using face recognition and sent notification to Telegram",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)