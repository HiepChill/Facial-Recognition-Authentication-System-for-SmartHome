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
from app.config import DATASET_DIR
from app.database import (
    setup_database, check_user_exists, add_user, add_face_image, 
    get_all_users, get_user_face_images, get_user_face_data, get_user_details,
    update_user, delete_user, delete_face_image
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_database
    setup_database()
    

app = FastAPI(
    title="Smart Home Security System", 
    lifespan=lifespan, 
    description="Hệ thống an ninh nhà thông minh sử dụng nhận diện khuôn mặt và thông báo Telegram."
)

app = FastAPI(
    title="Smart Home Security System", 
    lifespan=lifespan, 
    description="Hệ thống an ninh nhà thông minh sử dụng nhận diện khuôn mặt và thông báo Telegram."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/users")
async def list_users():
    users = get_all_users()
    return {"status": "success", "users": users, "total": len(users)}

@app.get("/users/{user_id}")
async def get_user(user_id: str = Path(...)):
    user = get_user_details(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    return {"status": "success", "user": user}

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
            faces = face_analyzer.get(img)
            if not faces:
                results.append({"filename": face_image.filename, "status": "error", "message": "Không tìm thấy khuôn mặt"})
                continue
            
            if len(faces) > 1:
                max_area = 0
                max_face_idx = 0
                for i, face in enumerate(faces):
                    bbox = face.bbox.astype(int)
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        max_face_idx = i
                face = faces[max_face_idx]
                bbox = face.bbox.astype(int)
                left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
                padding = 50
                height, width = img.shape[:2]
                left = max(0, left - padding)
                top = max(0, top - padding)
                right = min(width, right + padding)
                bottom = min(height, bottom + padding)
                img = img[top:bottom, left:right]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
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

@app.get("/users/{user_id}/faces")
async def list_faces(user_id: str = Path(...)):
    if not check_user_exists(user_id):
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    face_images = get_user_face_images(user_id)
    return {"status": "success", "face_images": face_images, "total": len(face_images)}

@app.delete("/faces/{image_id}")
async def remove_face(image_id: int = Path(...)):
    success, message = delete_face_image(image_id)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    global face_database
    user_face_data = get_user_face_data()
    face_database = load_face_database(user_face_data)
    return {"status": "success", "message": message}



@app.get("/")
async def root():
    return {
        "name": "Smart Home Security System",
        "version": "1.0.0",
        "description": "Hệ thống an ninh nhà thông minh sử dụng nhận diện khuôn mặt và thông báo Telegram.",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=True)