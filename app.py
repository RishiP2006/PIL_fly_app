import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import re
from huggingface_hub import hf_hub_download, HfApi

# Live video support
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.title("🪰 Drosophila Gender Detection")
st.write("Select a model and upload an image or use live camera.")

HF_REPO_ID = "RishiPTrial/drosophila-models"

# List files in HF repo
@st.cache_data(show_spinner=False)
def list_hf_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
    except Exception as e:
        st.error(f"Error listing files in Hugging Face repo: {e}")
        return []
    # filter only model files
    return [f for f in files if f.lower().endswith((".pt", ".keras", ".h5", ".pth")) and not f.startswith(".")]

# Build model info
@st.cache_data(show_spinner=False)
def build_models_info():
    files = list_hf_models()
    info = {}
    for fname in files:
        input_size = 224
        if "inceptionv3" in fname.lower():
            input_size = 299
        if fname.lower().endswith(".pt"):
            info[fname] = {"type": "detection", "framework": "yolo"}
        elif fname.lower().endswith((".keras", ".h5")):
            info[fname] = {"type": "classification", "framework": "keras", "input_size": input_size}
        elif fname.lower() == "model_final.pth":
            info[fname] = {"type": "classification", "framework": "torch_custom", "input_size": input_size}
        elif fname.lower().endswith(".pth"):
            info[fname] = {"type": "classification", "framework": "torch", "input_size": input_size}
    return info

MODELS_INFO = build_models_info()

# Check availability of frameworks
for name, info in MODELS_INFO.items():
    fw = info.get("framework")
    if fw == "keras":
        try:
            import tensorflow as _tf  # just test availability
        except ImportError:
            st.warning(f"Model '{name}' requires TensorFlow/Keras but unavailable.")
    if fw == "yolo":
        try:
            from ultralytics import YOLO as _Y
        except ImportError:
            st.warning(f"Model '{name}' requires Ultralytics YOLO but unavailable.")
    if fw in ("torch", "torch_custom"):
        try:
            import torch as _t
        except ImportError:
            st.warning(f"Model '{name}' requires PyTorch but unavailable.")

# Helpers to load models lazily
def load_model_final_pth(local_path):
    import torch
    import torch.nn as nn
    from torchvision import models
    checkpoint = torch.load(local_path, map_location="cpu")
    # Extract state_dict
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # build ResNet18 binary classifier
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_model_from_hf(name, info):
    # Download file
    try:
        local_path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    except Exception as e:
        st.error(f"Error downloading {name}: {e}")
        return None

    fw = info.get("framework")
    # Keras: apply monkey-patch just before import
    if fw == "keras":
        try:
            import sys
            # Monkey-patch keras aliases so config referring to keras.src.models.functional works
            import tensorflow as tf
            # Alias modules
            sys.modules['keras'] = tf.keras
            sys.modules['keras.layers'] = tf.keras.layers
            sys.modules['keras.models'] = tf.keras.models
            sys.modules['keras.src.models.functional'] = tf.keras.models
            sys.modules['keras.src.layers'] = tf.keras.layers
            # Patch InputLayer to accept batch_shape in config
            from tensorflow.keras.layers import InputLayer
            _orig_init = InputLayer.__init__
            def _patched_init(self, *args, batch_shape=None, **kwargs):
                if batch_shape is not None:
                    kwargs['batch_input_shape'] = tuple(batch_shape)
                return _orig_init(self, *args, **kwargs)
            InputLayer.__init__ = _patched_init

            # Now load the model
            model = tf.keras.models.load_model(local_path)
            return model
        except Exception as e:
            st.error(f"Failed loading Keras model {name}: {e}")
            return None

    # PyTorch custom
    if fw == "torch_custom":
        try:
            return load_model_final_pth(local_path)
        except Exception as e:
            st.error(f"Failed loading custom PyTorch model {name}: {e}")
            return None

    # Plain PyTorch
    if fw == "torch":
        try:
            import torch
            model = torch.load(local_path, map_location="cpu")
            model.eval()
            return model
        except Exception as e:
            st.error(f"Failed loading PyTorch model {name}: {e}")
            return None

    # YOLO
    if fw == "yolo":
        try:
            from ultralytics import YOLO
            return YOLO(local_path)
        except Exception as e:
            st.error(f"Failed loading YOLO model {name}: {e}")
            return None

    st.error(f"Unsupported framework for {name}")
    return None

# Preprocess image for classification
def preprocess_image_pil(pil_img, size):
    arr = pil_img.resize((size, size))
    arr = np.asarray(arr).astype(np.float32) / 255.0
    return arr  # RGB normalized

# Classification inference
def classify(model, img_array):
    x = np.expand_dims(img_array, axis=0)  # (1,H,W,3)
    # Keras?
    import inspect
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return model.predict(x)
    except Exception:
        pass
    # PyTorch?
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                x_t = torch.tensor(x).permute(0,3,1,2).float()
                out = model(x_t)
                return out.cpu().numpy()
    except Exception:
        pass
    st.error("Unknown model type for classification.")
    return None

# Interpret binary output
def interpret_classification(preds):
    if preds is None:
        return None, None
    arr = np.asarray(preds)
    # shape (1,2)
    if arr.ndim == 2 and arr.shape[1] == 2:
        exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        idx = int(np.argmax(probs, axis=1)[0])
        label = ["Male","Female"][idx]
        return label, float(probs[0][idx])
    # shape (1,1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0][0])
        prob = 1/(1+np.exp(-val)) if val < 0 or val > 1 else val
        label = "Female" if prob >= 0.5 else "Male"
        return label, (prob if label=="Female" else 1-prob)
    st.warning(f"Unexpected prediction shape: {arr.shape}")
    return None, None

# YOLO detection inference
def detect_yolo(model, pil_img):
    arr = np.array(pil_img.convert("RGB"))
    try:
        results = model.predict(source=arr)
    except Exception as e:
        st.error(f"YOLO inference failed: {e}")
        return []
    detections = []
    for res in results:
        for b in res.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            box = tuple(map(int, b.xyxy[0].cpu().numpy())) if hasattr(b.xyxy[0], 'cpu') else tuple(map(int, b.xyxy[0]))
            name = model.names.get(cls, str(cls)) if hasattr(model, 'names') else str(cls)
            detections.append((name, conf, box))
    return detections

# Video processor
class GenderDetectionProcessor(VideoProcessorBase):
    def __init__(self, model, info):
        self.model = model
        self.info = info
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if self.model is not None:
            if self.info.get("type") == "classification":
                size = self.info.get("input_size", 224)
                arr = preprocess_image_pil(pil, size)
                preds = classify(self.model, arr)
                label, prob = interpret_classification(preds)
                if label:
                    draw.text((10,10), f"{label} ({prob:.1%})", fill="red")
            else:
                dets = detect_yolo(self.model, pil)
                for name, conf, (x1,y1,x2,y2) in dets:
                    draw.rectangle([x1,y1,x2,y2], outline="green", width=2)
                    draw.text((x1, max(y1-10,0)), f"{name} {conf:.2f}", fill="green")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# UI: select model
safe_map = {re.sub(r"[^\w\s.-]","_", name): name for name in MODELS_INFO}
safe_names = list(safe_map.keys())
choice = st.selectbox("Select model", safe_names) if safe_names else None
model_name = safe_map.get(choice)
model = load_model_from_hf(model_name, MODELS_INFO[model_name]) if model_name else None

# Upload Image section
st.markdown("---")
st.subheader("📷 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if img_file and model is not None:
    pil_img = Image.open(img_file).convert("RGB")
    st.image(pil_img, use_column_width=True)
    info = MODELS_INFO[model_name]
    if info.get("type") == "classification":
        size = info.get("input_size", 224)
        arr = preprocess_image_pil(pil_img, size)
        preds = classify(model, arr)
        label, prob = interpret_classification(preds)
        if label:
            st.success(f"Prediction: {label} ({prob:.1%})")
    else:
        disp = pil_img.copy()
        draw = ImageDraw.Draw(disp)
        dets = detect_yolo(model, pil_img)
        male_count = female_count = 0
        for name, conf, box in dets:
            x1,y1,x2,y2 = box
            draw.rectangle([x1,y1,x2,y2], outline="green", width=2)
            draw.text((x1, max(y1-10,0)), f"{name} {conf:.2f}", fill="green")
            if name.lower()=="male":
                male_count += 1
            elif name.lower()=="female":
                female_count += 1
        st.image(disp, use_column_width=True)
        st.info(f"Detected Males: {male_count}, Females: {female_count}")

# Live camera
st.markdown("---")
st.subheader("📸 Live Camera Gender Detection")
if model is not None:
    ctx = webrtc_streamer(
        key="live-gender-detect",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: GenderDetectionProcessor(model, MODELS_INFO[model_name]),
        async_processing=True,
    )
else:
    st.warning("Please select a model first.")


