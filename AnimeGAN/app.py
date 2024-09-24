import os
import gradio as gr
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
import cv2
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_name = ort.get_device()

if device_name == 'cpu':
    providers = ['CPUExecutionProvider']
elif device_name == 'GPU':
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

#load model
mtcnn = MTCNN(image_size=256, margin=0, min_face_size=128, thresholds=[0.7, 0.8, 0.9], device=device)

# MTCNN for face detection with landmarks
def detect(img):
    # Detect faces
    batch_boxes, batch_probs, batch_points = mtcnn.detect(img, landmarks=True)
    return batch_boxes, batch_points


# Expand the area around the detected face by margin {ratio} pixels
def margin_face(box, img_HW, margin=0.5):
    x1, y1, x2, y2 = [c for c in box]
    w, h = x2 - x1, y2 - y1
    new_x1 = max(0, x1 - margin*w)
    new_x2 = min(img_HW[1], x2 + margin * w)
    x_d = min(x1-new_x1, new_x2-x2)
    new_w = x2 -x1 + 2 * x_d  
    new_x1 = x1-x_d
    new_x2 = x2+x_d

    # new_h = 1.25 * new_w   
    new_h = 1.0 * new_w   

    if new_h>=h:
        y_d = new_h-h  
        new_y1 = max(0, y1 - y_d//2)
        new_y2 = min(img_HW[0], y2 + y_d//2)
    else:
        y_d = abs(new_h - h) 
        new_y1 = max(0, y1 + y_d // 2)
        new_y2 = min(img_HW[0], y2 - y_d // 2)
    return list(map(int, [new_x1, new_y1, new_x2, new_y2]))

def process_image(img, x32=True):
    h, w = img.shape[:2]
    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/ 127.5 - 1.0
    return img

def load_image(image_path, focus_face):
    img0 = cv2.imread(image_path).astype(np.float32)
    if focus_face == "Yes":
        batch_boxes, batch_points = detect(img0)
        if batch_boxes is None:
            print("No face detected !")
            return
        [x1, y1, x2, y2] = margin_face(batch_boxes[0], img0.shape[:2])
        img0 = img0[y1:y2, x1:x2]
    img = process_image(img0)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]

def convert(img, model, scale):
    session = ort.InferenceSession(MODEL_PATH[model], providers=providers)
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name
    fake_img = session.run(None, {x : img})[0]
    images = (np.squeeze(fake_img) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    output_image = cv2.resize(images, (scale[1],scale[0]))
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    
os.makedirs('output', exist_ok=True)

MODEL_PATH = {
    "Hayao": './model/AnimeGANv2_Hayao.onnx',
    "Shinkai": './model/AnimeGANv2_Shinkai.onnx',
    "Paprika": './model/AnimeGANv2_Paprika.onnx',
    "PortraitSketch": './model/AnimeGANv3_PortraitSketch.onnx',
    "JP_face": './model/AnimeGANv3_JP_face.onnx',
}


def inference(upload, webcam, model, focus_face=None):
    print(upload, webcam, model, focus_face)
    if upload is not None:
        img_path = upload
    elif upload is None and webcam is not None:
        img_path = webcam
    else:
        img_path = ""
    mat, scale = load_image(img_path, focus_face)
    output = convert(mat, model, scale)
    save_path = f"output/out.{img_path.rsplit('.')[-1]}"
    cv2.imwrite(save_path, output)
    return output, save_path

### Layout ###

title = "AnimeGANv2: To produce your own animation 😶‍🌫️"
description = r"""
### 🔥Demo AnimeGANv2: To produce your own animation. <br>
#### How to use:
1a. Upload your image
1b. Use webcam to take an image
2. Select the style (**For human**: PortraitSketch, JP_face; **For scene**: Hayao, Shinkai, Paprika)
3. Choice of whether to extract the face.(Warning: Yes if there is a face in the image)
"""
article = r"""
<center><img src='https://visitor-badge.glitch.me/badge?page_id=AnimeGAN_demo&left_color=green&right_color=blue' alt='visitor badge'></center>
<center><a href='https://github.com/TachibanaYoshino/AnimeGANv3' target='_blank'>Github Repo</a></center>
"""
examples=[['sample1.jpg', None, 'PortraitSketch', "Yes"], 
          ['sample2.jpg', None, 'PortraitSketch', "No"],
          ['sample3.jpg', None, 'Hayao', "No"], 
          ['sample4.jpeg', None, 'Shinkai', "No"],
          ['sample5.jpg', None, 'Paprika', "No"], 
          ['sample6.jpeg', None, 'JP_face', "No"]]
gr.Interface(
    inference, [
        gr.Image(type="filepath", label="Image"),
        gr.Image(type="filepath", label="Webcam"),
        gr.Dropdown([
            'Hayao',
            'Shinkai',
            'Paprika',
            'PortraitSketch',
            'JP_face',
        ], 
            type="value",
            value='PortraitSketch',
            label='AnimeGAN Style'),
        gr.Radio(['Yes', 'No'], type="value", value='No', label='Extract face'),
    ], [
        gr.Image(type="numpy", label="Output (The whole image)"),
        gr.File(label="Download the output image")
    ],
    title=title,
    description=description,
    article=article,
    cache_examples=True,
    examples=examples,
    allow_flagging="never").launch(enable_queue=True)