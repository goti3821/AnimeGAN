import os
import cv2
import numpy as np
import onnxruntime as ort
from facenet_pytorch import MTCNN
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = ort.get_device()

if device_name == 'cpu':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Load MTCNN for face detection
mtcnn = MTCNN(image_size=256, margin=0, min_face_size=128, thresholds=[0.7, 0.8, 0.9], device=device)

def detect(img):
    batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
    return batch_boxes, batch_points

def margin_face(box, img_HW, margin=0.5):
    x1, y1, x2, y2 = [c for c in box]
    w, h = x2 - x1, y2 - y1
    new_x1 = max(0, x1 - margin * w)
    new_x2 = min(img_HW[1], x2 + margin * w)
    x_d = min(x1 - new_x1, new_x2 - x2)
    new_w = x2 - x1 + 2 * x_d  
    new_x1 = x1 - x_d
    new_x2 = x2 + x_d

    new_h = 1.0 * new_w    

    if new_h >= h:
        y_d = new_h - h  
        new_y1 = max(0, y1 - y_d // 2)
        new_y2 = min(img_HW[0], y2 + y_d // 2)
    else:
        y_d = abs(new_h - h) 
        new_y1 = max(0, y1 + y_d // 2)
        new_y2 = min(img_HW[0], y2 - y_d // 2)
    return list(map(int, [new_x1, new_y1, new_x2, new_y2]))

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32:  # Resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img

def load_image(image_path, focus_face):
    img0 = cv2.imread(image_path).astype(np.float32)
    if focus_face == "Yes":
        batch_boxes, batch_points = detect(img0)
        if batch_boxes is None:
            print("No face detected!")
            return None, None
        [x1, y1, x2, y2] = margin_face(batch_boxes[0], img0.shape[:2])
        img0 = img0[y1:y2, x1:x2]
    img = process_image(img0)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]

def convert(img, model):
    session = ort.InferenceSession(MODEL_PATH[model], providers=providers)
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name
    fake_img = session.run(None, {x: img})[0]
    images = (np.squeeze(fake_img) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    return images

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

MODEL_PATH = {
    "Hayao": './model/AnimeGANv2_Hayao.onnx',
    "Shinkai": './model/AnimeGANv2_Shinkai.onnx',
    "Paprika": './model/AnimeGANv2_Paprika.onnx',
    "PortraitSketch": './model/AnimeGANv3_PortraitSketch.onnx',
    "JP_face": './model/AnimeGANv3_JP_face.onnx',
}

def main():
    # Prompt for image file path
    image_path = input("Enter the path to the input image: ")
    
    # Prompt for model selection
    print("Select the style model:")
    for model_name in MODEL_PATH.keys():
        print(f"- {model_name}")
    selected_model = input("Enter the model name: ")

    # Prompt for focus face
    focus_face = input("Do you want to focus on the face? (Yes/No): ")

    # Load and process the image
    mat, scale = load_image(image_path, focus_face)
    if mat is None:
        return

    output = convert(mat, selected_model)
    output_image_path = f"output/out_{os.path.basename(image_path)}"
    cv2.imwrite(output_image_path, output)
    print(f"Output saved to: {output_image_path}")

if __name__ == "__main__":
    main()
