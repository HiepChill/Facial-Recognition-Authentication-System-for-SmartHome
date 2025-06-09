# Công Cụ Đánh Giá Mô Hình Nhận Diện Khuôn Mặt

Công cụ này cho phép đánh giá và so sánh hiệu suất của nhiều mô hình nhận diện khuôn mặt khác nhau.

## Các Mô Hình Được Hỗ Trợ

1. **InsightFace** - Buffalo_L model
2. **DeepFace** - Sử dụng FaceNet backend
3. **FaceNet-PyTorch** - Pre-trained trên VGGFace2
4. **Dlib** - ResNet-based face recognition
5. **FaceNet-TensorFlow** - Implementation gốc từ [davidsandberg/facenet](https://github.com/davidsandberg/facenet)

## Cài Đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Thiết lập các mô hình

#### InsightFace, DeepFace, FaceNet-PyTorch

Các mô hình này sẽ tự động tải xuống khi chạy lần đầu.

#### Dlib

Cần tải các file model sau:

-   `shape_predictor_68_face_landmarks.dat`
-   `dlib_face_recognition_resnet_model_v1.dat`

Đặt các file này trong thư mục `evaluation/`.

#### FaceNet-TensorFlow

Chạy script hướng dẫn:

```bash
python download_facenet_model.py
```

Hoặc tải thủ công:

1. Tải model từ: https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-
2. Giải nén và đặt file `.pb` vào `evaluation/facenet_models/20180402-114759.pb`

## Cấu Trúc Dữ Liệu

```
evaluation/
├── dataset/
│   ├── reference/          # Ảnh tham chiếu để tạo database
│   │   ├── user1/         # Mỗi người một thư mục
│   │   ├── user2/
│   │   └── ...
│   ├── known/             # Ảnh test của những người đã biết
│   │   ├── user1/
│   │   ├── user2/
│   │   └── ...
│   └── unknown/           # Ảnh của những người lạ
└── results/               # Kết quả đánh giá
```

## Sử Dụng

### 1. Chuẩn bị dữ liệu

Chạy để tạo cấu trúc thư mục mẫu:

```bash
python evaluate_models.py
# Chọn option 2: "Tạo cấu trúc thư mục mẫu"
```

Sau đó:

-   Đặt ảnh tham chiếu vào thư mục `dataset/reference/[tên_người]/`
-   Đặt ảnh test vào thư mục `dataset/known/[tên_người]/`
-   Đặt ảnh người lạ vào thư mục `dataset/unknown/`

### 2. Chạy đánh giá

```bash
python evaluate_models.py
# Chọn option 1: "Đánh giá các mô hình"
```

### 3. Xem kết quả

Kết quả sẽ được hiển thị dưới dạng bảng và lưu vào file JSON trong thư mục `results/`.

## Các Chỉ Số Đánh Giá

-   **Accuracy**: Tỷ lệ dự đoán đúng
-   **Precision**: Tỷ lệ true positive trong các dự đoán positive
-   **Recall**: Tỷ lệ true positive được phát hiện
-   **F1-Score**: Trung bình điều hòa của Precision và Recall
-   **Processing Time**: Thời gian xử lý trung bình cho mỗi ảnh
-   **Initialization Time**: Thời gian khởi tạo mô hình

## Lưu Ý

1. **Chất lượng ảnh**: Sử dụng ảnh có chất lượng tốt, độ phân giải cao
2. **Đa dạng**: Ảnh tham chiếu nên đa dạng về góc chụp, ánh sáng
3. **Số lượng**: Mỗi người nên có ít nhất 3-5 ảnh tham chiếu
4. **Định dạng**: Hỗ trợ JPG, JPEG, PNG
5. **Tên thư mục**: Tên thư mục trong `known/` phải giống với `reference/`

## Khắc Phục Sự Cố

### Lỗi thiếu model files

-   Dlib: Tải các file `.dat` cần thiết
-   FaceNet-TensorFlow: Chạy `python download_facenet_model.py`

### Lỗi memory

-   Giảm số lượng ảnh test
-   Sử dụng ảnh có độ phân giải thấp hơn

### Lỗi dependencies

```bash
pip install tensorflow deepface facenet-pytorch dlib scipy tqdm tabulate
```

## Tùy Chỉnh

### Thay đổi ngưỡng nhận diện

Chỉnh sửa biến `RECOGNITION_THRESHOLD` trong `evaluate_models.py` hoặc nhập khi chạy chương trình.

### Thêm mô hình mới

1. Tạo class kế thừa `FaceRecognitionModel`
2. Implement các method: `initialize()`, `get_embedding()`, `get_name()`
3. Thêm vào danh sách `models` trong hàm `main()`

## Performance Benchmark

Kết quả test trên LFW dataset:

| Model              | LFW Accuracy | Architecture        | Notes               |
| ------------------ | ------------ | ------------------- | ------------------- |
| FaceNet-TensorFlow | 0.9965       | Inception ResNet v1 | VGGFace2 pretrained |
| InsightFace        | ~0.995       | ArcFace             | Buffalo_L model     |
| FaceNet-PyTorch    | ~0.995       | Inception ResNet v1 | VGGFace2 pretrained |
| DeepFace           | ~0.97        | Various backends    | Configurable        |
| Dlib               | ~0.99        | ResNet              | Classical approach  |

## Tài Liệu Tham Khảo

-   [InsightFace](https://github.com/deepinsight/insightface)
-   [DeepFace](https://github.com/serengil/deepface)
-   [FaceNet-PyTorch](https://github.com/timesler/facenet-pytorch)
-   [Dlib](http://dlib.net/)
-   [FaceNet-TensorFlow](https://github.com/davidsandberg/facenet)
