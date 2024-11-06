import streamlit as st
import torch
import gc
import torch.cuda
from PIL import Image
import time
import os
from pathlib import Path
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torchvision.models import resnet50
import torchvision.transforms as transforms
from ultralytics import YOLO
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background: linear-gradient(to right, #00B4DB, #0083B0);
        color: white;
        border: none;
        font-weight: bold;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .metric-container {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    .main-header {
        font-family: 'SF Pro Display', sans-serif;
        font-weight: bold;
        font-size: 2.5em;
        background: linear-gradient(45deg, #00B4DB, #0083B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1em;
    }
</style>
""", unsafe_allow_html=True)


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# Model class definitions
class FusionModel(torch.nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, hidden_dim):
        super(FusionModel, self).__init__()
        self.fc = torch.nn.Linear(image_embedding_dim + text_embedding_dim, hidden_dim)
    
    def forward(self, image_embeds, text_embeds):
        fused_embeds = torch.cat((image_embeds, text_embeds), dim=1)
        hidden = torch.relu(self.fc(fused_embeds))
        return hidden
    
class LateFusionModel(torch.nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, hidden_dim):
        super(LateFusionModel, self).__init__()
        self.image_fc = torch.nn.Linear(image_embedding_dim, hidden_dim)  # Fully connected layer for image embeddings
        self.text_fc = torch.nn.Linear(text_embedding_dim, hidden_dim)  # Fully connected layer for text embeddings
        self.fc_out = torch.nn.Linear(hidden_dim, hidden_dim)  # Final fully connected layer after fusion

    def forward(self, image_embeds, text_embeds):
        # Process the image and text embeddings separately
        image_hidden = torch.relu(self.image_fc(image_embeds))
        text_hidden = torch.relu(self.text_fc(text_embeds))
        
        # Combine (fuse) the two processed embeddings at a later stage
        fused_hidden = image_hidden + text_hidden  # Late fusion: adding processed embeddings
        
        return fused_hidden

# Transform pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_models(model_path_fusion, model_path_t5, model_hidden_dim,fusion_type,t5_model):
    clear_gpu_memory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load all required models
    yolo_model = YOLO("yolov8n.pt")
    
    resnet = resnet50(pretrained=True).to(device)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert.eval()
    
    t5_tokenizer = T5Tokenizer.from_pretrained(f't5-{t5_model}')
    t5_model = T5ForConditionalGeneration.from_pretrained(f't5-{t5_model}').to(device)
    
    t5_model.load_state_dict(torch.load(model_path_t5,weights_only=True))
    t5_model.eval()  # Set to evaluation mode
    
    if fusion_type == 'early':
    
        fusion_model = FusionModel(image_embedding_dim=2048, text_embedding_dim=768, hidden_dim=model_hidden_dim)
    elif fusion_type == 'late':
        fusion_model = LateFusionModel(image_embedding_dim=2048, text_embedding_dim=768, hidden_dim=model_hidden_dim)
    
    
    fusion_model.load_state_dict(torch.load(model_path_fusion,weights_only=True))
    fusion_model.to(device)
    fusion_model.eval()
    
    
    
    
    return device, yolo_model, resnet, bert, bert_tokenizer, t5_model, t5_tokenizer, fusion_model

def generate_yolo_labels(image, yolo_model):
    results = yolo_model(image, verbose=False)
    detected_class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_model.names
    detected_objects = [class_names[i] for i in detected_class_indices]
    return " ".join(detected_objects)

def get_image_embedding(image, resnet, device):
    image = image.to(device)
    with torch.no_grad():
        image_embeds = resnet(image.unsqueeze(0)).squeeze()
    return image_embeds

def get_yolo_label_embedding(labels, bert_tokenizer, bert, device):
    inputs = bert_tokenizer(labels, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        text_embeds = bert(**inputs).pooler_output
    return text_embeds.squeeze()

def generate_caption(image, fusion_model, t5_model, device, models):
    start_time = time.time()
    
    # Get YOLO labels
    yolo_start = time.time()
    yolo_labels = generate_yolo_labels(image, models[1])  # models[1] is yolo_model
    yolo_time = time.time() - yolo_start
    
    # Process image
    image_start = time.time()
    image_tensor = transform(image).to(device)
    image_embeds = get_image_embedding(image_tensor, models[2], device)  # models[2] is resnet
    image_time = time.time() - image_start
    
    # Get text embeddings
    text_start = time.time()
    text_embeds = get_yolo_label_embedding(yolo_labels, models[4], models[3], device)  # models[3] is bert, models[4] is bert_tokenizer
    text_time = time.time() - text_start
    
    # Fuse embeddings and generate caption
    caption_start = time.time()
    fused_embeds = fusion_model(image_embeds.unsqueeze(0), text_embeds.unsqueeze(0)).squeeze()
    fused_embeds = fused_embeds.unsqueeze(0).unsqueeze(1)
    
    encoder_outputs = BaseModelOutput(last_hidden_state=fused_embeds)
    generated_ids = t5_model.generate(encoder_outputs=encoder_outputs, max_length=50)
    generated_caption = models[6].decode(generated_ids[0], skip_special_tokens=True)  # models[6] is t5_tokenizer
    
    print(f'caption : {generated_caption}')
    
    caption_time = time.time() - caption_start
    
    total_time = time.time() - start_time
    
    timing_metrics = {
        'YOLO Detection': yolo_time,
        'Image Processing': image_time,
        'Text Embedding': text_time,
        'Caption Generation': caption_time,
        'Total Time': total_time
    }
    
    return generated_caption, yolo_labels, timing_metrics

def main():
    st.markdown('<h1 class="main-header">🖼️ AI Image Caption Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Model Configuration")
        model_folders = [d for d in os.listdir('mmai-models') if os.path.isdir(os.path.join('mmai-models', d))]
        selected_model = st.selectbox(
            "Select Model Architecture",
            model_folders,
            format_func=lambda x: x.replace('-', ' ').title()
        )
        
        st.markdown("---")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🖥️ Running on: {device}")
        st.markdown(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file...",
            type=["jpg", "jpeg", "png"]
        )
    
    if uploaded_file:
        # Load models
        model_path_fusion = os.path.join('mmai-models', selected_model, 'fusion_model.pth')
        model_path_t5 = os.path.join('mmai-models', selected_model, 't5_model.pth')
        hidden_dim = {"small": 512, "base": 768, "large": 1024}
        model_hidden_dim = hidden_dim[selected_model.split('-')[2]]
        fusion_type = selected_model.split('-')[3]
        t5_model = selected_model.split('-')[2]
        print(model_path_fusion, model_path_t5, model_hidden_dim,fusion_type,t5_model)
        
        
        
        with st.spinner("🔄 Loading models..."):
            models = load_models(model_path_fusion, model_path_t5, model_hidden_dim,fusion_type,t5_model)
        
        
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, use_column_width=True, caption="Uploaded Image")
        
        if st.button("🎯 Generate Caption"):
            with st.spinner("🤖 Analyzing image..."):
                caption, objects, metrics = generate_caption(
                    image, 
                    models[7],  # fusion_model
                    models[5],  # t5_model
                    models[0],  # device
                    models
                )
            
            with col2:
                st.markdown("### Analysis Results")
                
                # Caption
                st.markdown(f"""
                    <div class="metric-container">
                        <h4>📝 Generated Caption</h4>
                        <p style="font-size: 1.2em;">{caption}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Detected objects
                st.markdown(f"""
                    <div class="metric-container">
                        <h4>🔍 Detected Objects</h4>
                        <p>{objects}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Performance metrics
                st.markdown("### ⚡ Performance Metrics")
                fig = go.Figure(go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=['#00B4DB', '#0083B0', '#006C8F', '#005571', '#004254'],
                    text=[f'{v*1000:.1f}ms' for v in metrics.values()],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Processing Time Breakdown",
                    xaxis_title="Stage",
                    yaxis_title="Time (seconds)",
                    template="plotly_dark" if st.sidebar.checkbox("Dark Mode") else "plotly_white",
                    height=300,
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                
                
        if st.session_state.get('previous_model') != selected_model:
            clear_gpu_memory()
            st.session_state['previous_model'] = selected_model

if __name__ == "__main__":
    main()