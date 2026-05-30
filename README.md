# Phần mềm giám sát an ninh và cảnh báo xâm nhập

Dự án sử dụng **YOLOv8** để phát hiện người trong khung hình camera/video, sau đó kiểm tra xem tâm của bounding box người đó có nằm trong vùng giám sát **ROI Polygon** hay không. Nếu người được phát hiện nằm trong vùng đã chỉ định, hệ thống sẽ hiển thị cảnh báo xâm nhập.

---

## 1. Chức năng chính

- Mở camera bằng OpenCV.
- Cho phép người dùng tự vẽ vùng giám sát dạng đa giác bằng chuột.
- Nhấn phím để khóa vùng ROI.
- Load model YOLO đã train từ file `best.pt`.
- Phát hiện đối tượng thuộc class `person`.
- Lọc kết quả dựa trên ngưỡng confidence.
- Tính tâm bounding box của người được phát hiện.
- Kiểm tra tâm bounding box có nằm trong ROI hay không.
- Hiển thị cảnh báo khi phát hiện người trong vùng chỉ định.

---

## 2. Cấu trúc thư mục đề xuất

```text
nhandien/
├── app.py
├── README.md
├── roi_points.json          # tự tạo sau khi khóa ROI
├── Data/
│   └── best.pt              # model YOLO đã train
└── Dataset/                 # chỉ cần nếu muốn train/fine-tune lại
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

Lưu ý: file `best.pt` phải nằm đúng đường dẫn:

```text
Data/best.pt
```

Vì trong code đang load model bằng:

```python
model = YOLO("Data/best.pt")
```

---

## 3. Cài đặt môi trường

### Bước 1: Tạo môi trường ảo

```bash
python -m venv venv
```

### Bước 2: Kích hoạt môi trường ảo

Trên Windows:

```bash
venv\Scripts\activate
```

### Bước 3: Cài thư viện cần thiết

```bash
pip install ultralytics opencv-python numpy
```
---

## 4. Chạy chương trình

Mở terminal tại thư mục chứa `app.py`, sau đó chạy:

```bash
python app.py
```

Nếu chạy từ thư mục cha, cần đảm bảo đường dẫn đến model `Data/best.pt` vẫn đúng.

---

## 5. Hướng dẫn sử dụng

Sau khi chạy chương trình, cửa sổ camera sẽ hiện lên.

Các phím điều khiển:

```text
Click chuột trái  : thêm điểm ROI
Click chuột phải : xóa điểm ROI cuối cùng
S                : khóa vùng ROI
R                : vẽ lại vùng ROI
P                : in tọa độ ROI ra terminal
Q                : thoát chương trình
```

Quy trình sử dụng:

```text
1. Chạy app.py.
2. Click chuột trái để chọn các điểm tạo vùng giám sát.
3. Chọn ít nhất 3 điểm để tạo thành polygon.
4. Nhấn S để khóa vùng ROI.
5. Khi có người đi vào vùng ROI, hệ thống sẽ hiển thị cảnh báo.
6. Nhấn R nếu muốn vẽ lại vùng.
7. Nhấn Q để thoát.
```

---

## 6. Cách hoạt động của thuật toán

### Bước 1: Đọc frame từ camera

Chương trình dùng OpenCV để đọc hình ảnh từ camera:

```python
cap = cv2.VideoCapture(VIDEO_SOURCE)
```

Camera được thiết lập kích thước frame:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

---

### Bước 2: Vẽ vùng ROI

Người dùng click chuột để tạo danh sách điểm polygon:

```python
polygon_points = []
```

Khi nhấn `S`, vùng ROI được khóa và lưu vào file:

```text
roi_points.json
```

---

### Bước 3: Detect người bằng YOLO

Model được load từ file:

```python
model = YOLO("Data/best.pt")
```

Mỗi vài frame, chương trình chạy YOLO để giảm giật:

```python
DETECT_EVERY_N_FRAMES = 5
```

Chỉ những object có class là `person` và confidence đủ cao mới được xử lý:

```python
CONF_THRESHOLD = 0.75
PERSON_CLASS_ID = 0
```

---

### Bước 4: Tính tâm bounding box

Sau khi YOLO phát hiện người, chương trình lấy tọa độ bounding box:

```python
x1, y1, x2, y2 = map(int, box.xyxy[0])
```

Sau đó tính tâm bounding box:

```python
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2
center_point = (center_x, center_y)
```

---

### Bước 5: Kiểm tra tâm bounding box trong ROI

Chương trình dùng hàm `cv2.pointPolygonTest()` để kiểm tra tâm bounding box có nằm trong polygon hay không:

```python
inside_roi = is_point_inside_polygon(center_point, polygon_points)
```

Nếu tâm bounding box nằm trong ROI:

```text
inside_roi = True
```

thì hệ thống xác nhận xâm nhập.

---

### Bước 6: Hiển thị cảnh báo

Nếu phát hiện người trong vùng chỉ định, chương trình hiển thị:

```text
CANH BAO XAM NHAP!
```

Bounding box sẽ có màu đỏ và label là:

```text
INTRUSION
```

Nếu người ở ngoài vùng ROI, bounding box sẽ có màu xanh và label là:

```text
PERSON
```

---

## 7. Lưu ý về model

File `best.pt` là model YOLO đã được train trước. Khi chạy app, không cần toàn bộ dataset train/val. Dataset chỉ cần khi muốn train lại hoặc fine-tune model.

Để chạy chương trình, chỉ cần:

```text
app.py
Data/best.pt
các thư viện Python cần thiết
```

---

## 8. Một số lỗi thường gặp

### Lỗi không tìm thấy `best.pt`

Nếu gặp lỗi:

```text
FileNotFoundError: No such file or directory: 'Data/best.pt'
```

Kiểm tra lại file model có nằm đúng vị trí không:

```text
nhandien/Data/best.pt
```

Nếu model nằm nơi khác, sửa đường dẫn trong code:

```python
model = YOLO("đường_dẫn_tới_best.pt")
```

---

### Model nhận nhầm áo khoác/móc đồ là người

Đây là hạn chế của model nếu dữ liệu train chưa đủ đa dạng. Cách cải thiện:

- Tăng thêm ảnh áo khoác, balo, ghế, móc đồ vào dataset.
- Các ảnh này nên để label rỗng nếu không có người.
- Fine-tune lại model từ `best.pt`.
- Khi demo, nên chọn vùng ROI sạch, ít vật thể giống người.

---

## 9. Fine-tune model nếu cần

Nếu có thêm dataset và muốn train tiếp từ model hiện tại:

```bash
yolo detect train data=data.yaml model=Data/best.pt epochs=10 imgsz=224 batch=4
```

Sau khi train xong, model mới thường nằm ở:

```text
runs/detect/train/weights/best.pt
```

Copy file này thay cho:

```text
Data/best.pt
```

---

## 10. Giới hạn hiện tại

- Hệ thống phụ thuộc vào độ chính xác của model YOLO.
- Nếu vật thể giống người bị nhận nhầm là `person`, hệ thống có thể cảnh báo sai.
- Ảnh đầu vào kích thước nhỏ có thể làm giảm độ chính xác.
- Hệ thống hiện kiểm tra bằng tâm bounding box, chưa dùng tracking nhiều người phức tạp.
- Vùng ROI được lưu ra file JSON nhưng phiên bản hiện tại chưa tự load lại ROI khi mở app lần sau.

---

## 11. Kết luận

Chương trình đã thực hiện được yêu cầu chính của đề tài: phát hiện người trong khung hình camera, kiểm tra vị trí người so với vùng giám sát do người dùng chỉ định, và phát cảnh báo khi người xuất hiện trong vùng ROI. Đây là nền tảng để phát triển một phần mềm giám sát an ninh và cảnh báo xâm nhập đơn giản.
