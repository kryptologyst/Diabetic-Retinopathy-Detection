"""Streamlit demo application for diabetic retinopathy detection."""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import logging

# Configure page
st.set_page_config(
    page_title="Diabetic Retinopathy Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .no-dr {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .dr {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer banner
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research demonstration tool only.</strong></p>
    <ul>
        <li>This application is NOT intended for clinical diagnosis or medical decision-making</li>
        <li>Results should NOT be used as a substitute for professional medical advice</li>
        <li>Always consult with qualified healthcare professionals for medical decisions</li>
        <li>This tool is for educational and research purposes only</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_options = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet-18": "resnet18",
    "Vision Transformer": "vit_base_patch16_224",
    "Attention EfficientNet": "attention_efficientnet",
    "Multi-scale EfficientNet": "multiscale_efficientnet",
    "Ensemble": "ensemble"
}

selected_model = st.sidebar.selectbox(
    "Select Model Architecture",
    list(model_options.keys()),
    index=0
)

model_name = model_options[selected_model]

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Minimum confidence required for DR prediction"
)

# Show attention maps
show_attention = st.sidebar.checkbox(
    "Show Attention Maps",
    value=True,
    help="Display attention/heatmap overlays"
)

# Load model function
@st.cache_resource
def load_model(model_name: str):
    """Load the specified model."""
    try:
        # Load configuration
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default configuration
            config = {
                'model': {'name': model_name, 'num_classes': 2, 'pretrained': True, 'dropout': 0.2},
                'data': {'num_classes': 2, 'class_names': ['No DR', 'DR'], 'input_size': [224, 224]},
                'device': {'fallback_order': ['cuda', 'mps', 'cpu']}
            }
        
        # Import model creation function
        from src.models.models import create_model
        
        # Create model
        model = create_model(config)
        
        # Load weights if available
        model_path = Path(f"outputs/best_model.pth")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            st.sidebar.success(f"✅ Loaded trained model from {model_path}")
        else:
            st.sidebar.warning("⚠️ No trained model found. Using pretrained weights only.")
        
        model.eval()
        return model, config
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, config = load_model(model_name)

if model is None:
    st.error("Failed to load model. Please check the configuration.")
    st.stop()

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

st.sidebar.info(f"🖥️ Using device: {device}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Retinal Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a retinal fundus photograph for DR detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess image
        def preprocess_image(image):
            """Preprocess image for model input."""
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            input_size = tuple(config['data']['input_size'])
            image = image.resize(input_size)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize
            image_array = image_array.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
            
            return image_tensor, image_array
        
        # Preprocess
        image_tensor, processed_image = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            model = model.to(device)
            
            if hasattr(model, 'backbone') and hasattr(model.backbone, 'features_only'):
                # Multi-scale or attention model
                if model_name == "attention_efficientnet":
                    logits, attention_map = model(image_tensor)
                else:
                    logits = model(image_tensor)
            else:
                logits = model(image_tensor)
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=1)
        confidence = probabilities.max().item()
        prediction = probabilities.argmax(dim=1).item()
        
        # Convert to numpy for visualization
        probabilities_np = probabilities.cpu().numpy()[0]
        
        with col2:
            st.header("🔍 Prediction Results")
            
            # Prediction result
            class_names = config['data']['class_names']
            predicted_class = class_names[prediction]
            
            if prediction == 0:  # No DR
                st.markdown(f'''
                <div class="prediction-result no-dr">
                    ✅ {predicted_class}<br>
                    Confidence: {confidence:.1%}
                </div>
                ''', unsafe_allow_html=True)
            else:  # DR
                st.markdown(f'''
                <div class="prediction-result dr">
                    ⚠️ {predicted_class}<br>
                    Confidence: {confidence:.1%}
                </div>
                ''', unsafe_allow_html=True)
            
            # Confidence threshold warning
            if confidence < confidence_threshold:
                st.warning(f"⚠️ Low confidence prediction ({confidence:.1%}). Consider consulting a specialist.")
            
            # Probability distribution
            st.subheader("📊 Probability Distribution")
            
            fig = px.bar(
                x=class_names,
                y=probabilities_np,
                title="Class Probabilities",
                color=probabilities_np,
                color_continuous_scale=['green', 'red']
            )
            fig.update_layout(
                xaxis_title="Class",
                yaxis_title="Probability",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.subheader("📈 Detailed Metrics")
            
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                st.metric(
                    "No DR Probability",
                    f"{probabilities_np[0]:.1%}",
                    delta=None
                )
            
            with col_metric2:
                st.metric(
                    "DR Probability",
                    f"{probabilities_np[1]:.1%}",
                    delta=None
                )
            
            # Attention maps (if available)
            if show_attention and hasattr(model, 'backbone') and model_name == "attention_efficientnet":
                st.subheader("🎯 Attention Map")
                
                try:
                    # Get attention map
                    attention_map = attention_map.cpu().numpy()[0, 0]  # Remove batch and channel dims
                    
                    # Resize attention map to original image size
                    attention_resized = cv2.resize(attention_map, (224, 224))
                    
                    # Create overlay
                    original_image = np.array(image.resize((224, 224)))
                    overlay = original_image.copy()
                    
                    # Apply attention map as heatmap
                    attention_colored = cv2.applyColorMap(
                        (attention_resized * 255).astype(np.uint8),
                        cv2.COLORMAP_JET
                    )
                    
                    # Blend with original image
                    overlay = cv2.addWeighted(original_image, 0.6, attention_colored, 0.4, 0)
                    
                    st.image(overlay, caption="Attention Map Overlay", use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not generate attention map: {str(e)}")
            
            # Interpretation guide
            st.subheader("📋 Interpretation Guide")
            
            if prediction == 0:
                st.success("""
                **No Diabetic Retinopathy Detected**
                
                - The model did not detect signs of diabetic retinopathy
                - This does not guarantee absence of the condition
                - Regular eye exams are still recommended for diabetic patients
                """)
            else:
                st.error("""
                **Diabetic Retinopathy Detected**
                
                - The model detected potential signs of diabetic retinopathy
                - This requires immediate consultation with an eye care specialist
                - Early detection and treatment can prevent vision loss
                """)
            
            # Clinical recommendations
            st.subheader("🏥 Clinical Recommendations")
            
            st.info("""
            **For All Patients:**
            - Regular comprehensive eye exams are essential
            - Maintain good blood sugar control
            - Monitor blood pressure and cholesterol
            - Report any vision changes immediately
            
            **For Diabetic Patients:**
            - Annual dilated eye exams recommended
            - More frequent exams if DR is detected
            - Follow treatment plans as prescribed
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>This application is for research and educational purposes only.</p>
    <p>Not intended for clinical diagnosis or medical decision-making.</p>
    <p>Always consult with qualified healthcare professionals.</p>
</div>
""", unsafe_allow_html=True)
