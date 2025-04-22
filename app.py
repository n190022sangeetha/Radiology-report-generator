
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model Definition
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6 * 6 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformations for preprocessing the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Label dictionaries and model paths for body parts and diseases
label_dict = {
    0: "xrayctscan",
    1: "others"
}

label_dict1 = {
    0: "Bone_Fracture_Binary_Classification",
    1: "brain-tumor",
    2: "bone",
    3: "alzheimer_mri",
    4: "chest"
}

disease_model_paths = {
    "Bone_Fracture_Binary_Classification": "C:\\Users\\sangeetha\\bone_10",
    "brain-tumor": "C:\\Users\\sangeetha\\model_16",
    "bone": "C:\\Users\\sangeetha\\boneknee_20",
    "alzheimer_mri": "C:\\Users\\sangeetha\\modelalzhe_4",
    "chest": "C:\\Users\\sangeetha\\modelchest_19",
}

disease_labels = {
    "Bone_Fracture_Binary_Classification": ["fractured", "not fractured"],
    "brain-tumor": ["Glioma", "Pituitary", "Meningioma", "No Tumor", "Other"],
    "bone": ["Osteoporosis", "Normal", "Osteopenia"],
    "alzheimer_mri": ["Non Demented", "Mild Dementia", "Very mild Dementia", "Moderate Dementia"],
    "chest": ["TUBERCULOSIS", "NORMAL", "PNEUMONIA"],
}

# Preprocess image
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Check if the image is X-ray or CT Scan
def is_xray_or_ctscan(image_tensor):
    model = MyModel(num_classes=2).to(device)
    model_path = "C:\\Users\\sangeetha\\modelmain_3"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Predict body part
def predict_body_part(image_tensor):
    model = MyModel(num_classes=5).to(device)
    model_path = "C:\\Users\\sangeetha\\class_2"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return label_dict1[predicted.item()]

# Predict disease
def predict_disease(image_tensor, body_part):
    model_path = disease_model_paths[body_part]
    labels = disease_labels[body_part]
    model = MyModel(num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]



# Streamlit app interface
st.set_page_config(page_title="Radiology report generator", layout="centered")
st.title("🧠Radiology report generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Classifying..."):
        image_tensor = preprocess_image(uploaded_file)
        result = is_xray_or_ctscan(image_tensor)
        
        if result == 0:
            st.success("The image is identified as a valid X-ray or CT scan.")

            body_part = predict_body_part(image_tensor)
            disease = predict_disease(image_tensor, body_part)

            st.success(f"**Predicted Body Part:** {body_part}")
            st.success(f"**Predicted Disease:** {disease}")

            
            # Diagnostic Report
            report = f"""
            📋 **Diagnostic Report**
            
            • Scan Type: X-ray / CT  
            • Predicted Body Part: {body_part}  
            • Predicted Disease: {disease}  

            
            ⚠️ *Please consult a medical professional for confirmation.*
            """


            st.markdown(report, unsafe_allow_html=True)

            # Create report text including base64 image for download
            download_report = f"""
            📋 **Diagnostic Report**
            
            • Scan Type: X-ray / CT  
            • Predicted Body Part: {body_part}  
            • Predicted Disease: {disease}  
            
            ⚠️ *Please consult a medical professional for confirmation.* """

            st.download_button("📥 Download Report", download_report, file_name="diagnostic_report.txt", mime="text/plain")

        else:
            st.error("Please upload a valid X-ray or CT scan image. The uploaded image is classified as 'Other'.")
            st.stop()
