import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import timm # <-- Thêm import cho thư viện timm

# --- KHỞI TẠO ỨNG DỤNG FLASK ---
app = Flask(__name__)
CORS(app)

# --- ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH ---
# Sử dụng đúng kiến trúc pvt_v2_b4 từ thư viện timm
def create_model(num_classes=6):
    # Dòng này đã được cập nhật để sử dụng đúng mô hình của bạn
    model = timm.create_model(
        'pvt_v2_b4',       # <-- THAY ĐỔI QUAN TRỌNG Ở ĐÂY
        pretrained=False,  # Không dùng trọng số mặc định
        num_classes=num_classes
    )
    return model

# --- TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN ---
# Đường dẫn đến file .pth của bạn
MODEL_PATH = 'model/baseline_pvt_v2_b4_optimized.pth' # <-- NHỚ ĐỔI TÊN FILE CHO ĐÚNG
NUM_CLASSES = 6

# Tạo một thực thể của mô hình và tải trọng số
model = create_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# --- ĐỊNH NGHĨA CÁC LỚP BỆNH ---
class_names = {
    0: 'Bệnh Tảo Lá (Leaf Algal)',
    1: 'Bệnh Cháy Lá (Leaf Blight)',
    2: 'Bệnh Thán Thư (Colletotrichum)',
    3: 'Lá Khỏe Mạnh (Healthy)',
    4: 'Bệnh Đốm Lá Phomopsis (Phomopsis)',
    5: 'Bệnh Nấm Hồng (Rhizoctonia)'
}

# --- ĐỊNH NGHĨA QUÁ TRÌNH BIẾN ĐỔI ẢNH ---
# QUAN TRỌNG: Các thông số này phải khớp với lúc bạn huấn luyện mô hình.
# Hãy kiểm tra lại code training để đảm bảo các giá trị Resize, CenterCrop và Normalize là chính xác.
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

# --- TẠO API ENDPOINT CHO VIỆC DỰ ĐOÁN ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400
    
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            _, y_hat = torch.max(outputs, 1)
            predicted_idx = y_hat.item()
            predicted_class_name = class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{confidence:.2%}"
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'error during prediction'}), 500

# --- CHẠY SERVER ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)
