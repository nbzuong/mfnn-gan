import streamlit as st
import torch
from torchvision import transforms
from model import Pix2PixHDModel, InferenceModel
from options.train_options import TrainOptions
from PIL import Image
import numpy as np

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parse options and configure model
opt = TrainOptions().parse()
opt.no_instance = False
opt.continue_train = False
opt.load_pretrain = '/checkpoint/HKPU_1st/'
opt.isTrain = False

model = Pix2PixHDModel()
model.initialize(opt)
model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Inference function
def infer(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        residual_img = model.netR.forward(img)
        fake_image, _ = model.inference(img, residual_img)
        fake_image = fake_image.squeeze(0).cpu().numpy()
    return fake_image

# Streamlit app
st.title("Pix2PixHD Image Transformation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    fake_image = infer(image)

    # Convert fake_image to PIL format for displaying
    fake_image = np.transpose(fake_image, (1, 2, 0))  # CHW to HWC
    fake_image = (fake_image - fake_image.min()) / (fake_image.max() - fake_image.min())  # Normalize to [0, 1]
    fake_image = (fake_image * 255).astype(np.uint8)
    fake_image_pil = Image.fromarray(fake_image)

    # Display side by side images
    col1, col2 = st.columns(2)

    with col1:
        st.header("Uploaded Image")
        st.image(image, use_column_width=True)

    with col2:
        st.header("Generated Image")
        st.image(fake_image_pil, use_column_width=True)
