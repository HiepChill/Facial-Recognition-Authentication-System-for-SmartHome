import insightface
import cv2
import numpy as np

# Fix lỗi numpy deprecated np.int
np.int = int

#load facial recognition model (ArcFace)
model = insightface.app.FaceAnalysis(name="buffalo_l", providers=["DmlExecutionProvider"])
model.prepare(ctx_id=0)
 
#Read image
img = cv2.imread(r"./dataset/01_HiepChu/01_HiepChu_001.jpg")
print(img)
# img = cv2.resize(img, (640, 480))

#Face Detection
# faces = model.get(img)
# bbox = faces[0].bbox.astype(int)
# print(bbox)
# img = model.draw_on(img, faces)

# Hiển thị ảnh với các khuôn mặt đã được vẽ
# Sau khi vẽ khuôn mặt lên ảnh và trước khi hiển thị

# Resize ảnh để có thể hiển thị vừa màn hình
# Lấy kích thước màn hình
# screen_res = 1024, 768  # Kích thước màn hình mặc định, có thể điều chỉnh
# scale_width = screen_res[0] / img.shape[1]
# scale_height = screen_res[1] / img.shape[0]
# scale = min(scale_width, scale_height)

# # Nếu ảnh lớn hơn màn hình, resize xuống
# if scale < 1:
#     # Tính toán kích thước mới
#     window_width = int(img.shape[1] * scale)
#     window_height = int(img.shape[0] * scale)
#     dim = (window_width, window_height)
    
#     # Resize ảnh
#     img_display = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# else:
#     img_display = img.copy()

# # Hiển thị ảnh đã resize
# cv2.imshow('Face Detection Result', img_display)
# cv2.waitKey(0)  # Chờ nhấn phím bất kỳ
# cv2.destroyAllWindows()  # Đóng cửa sổ hiển thị

# print(faces)

# #If there is a face, get the embedding
# if faces:
#     face_embedding = faces[0].normed_embedding
#     print(f"Embedding vector (512-D): {face_embedding}")


# Khởi tạo mô hình FaceAnalysis với buffalo_l
# model = insightface.app.FaceAnalysis(name='buffalo_l')
# model.prepare(ctx_id=0)  # ctx_id=0 cho GPU, -1 cho CPU

# # Đọc ảnh
# img = cv2.imread(r"./dataset/01_HiepChu/01_HiepChu_001.jpg")
# img = cv2.resize(img, (640, 480))  # Thay đổi kích thước phù hợp

# # Phát hiện khuôn mặt
# faces = model.get(img)

# # Xử lý kết quả
# for face in faces:
#     bbox = face.bbox.astype(int)  # Hộp giới hạn
#     landmarks = face.landmark  # 5 điểm mốc
#     score = face.det_score  # Điểm số độ tin cậy
#     print(f"Bounding box: {bbox}, Score: {score}, Landmarks: {landmarks}")

#     # Vẽ hộp giới hạn và điểm mốc lên ảnh
#     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#     for (x, y) in landmarks:
#         cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

# # Lưu hoặc hiển thị ảnh
# cv2.imwrite('output.jpg', img)
