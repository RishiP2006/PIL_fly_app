import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import re
from huggingface_hub import hf_hub_download
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.title("Drosophila Gender Detection")
st.write("Select a model and upload an image or use live camera.")

HF_REPO_ID = "RishiPTrial/models_h5"

# Try importing ultralytics
try:
    import ultralytics
    from ultralytics import YOLO
    _ULTRA_AVAILABLE = True
except:
    _ULTRA_AVAILABLE = False

@st.cache_data(show_spinner=False)
def list_models():
    # Static list of known model files
    return [
        "drosophila_resnet50_classifier.h5",
        "drosophila_inceptionv3_classifier.h5",
        "mobilenetv2_classifier_final.h5",
        "best.pt",
        "model_final.pth",
    ]

@st.cache_data(show_spinner=False)
def build_model_info():
    info = {}
    for f in list_models():
        lower = f.lower()
        if lower.endswith(".pt"):
            info[f] = {"type": "detection", "framework": "yolo"}
        elif f == "model_final.pth":
            info[f] = {"type": "classification", "framework": "torch_custom", "input_size": 224}
        elif lower.endswith(".pth"):
            info[f] = {"type": "classification", "framework": "torch", "input_size": 224}
        elif lower.endswith(".h5"):
            size = 299 if "inceptionv3" in lower else 224
            info[f] = {"type": "classification", "framework": "keras", "input_size": size}
    return info

MODELS_INFO = build_model_info()
if not MODELS_INFO:
    st.error(f"No model files found in HF repo {HF_REPO_ID}")

@st.cache_resource(show_spinner=False)
def load_model(name, info):
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    except Exception as e:
        st.error(f"Error downloading {name}: {e}")
        return None

    fw = info["framework"]
    if fw == "keras":
        try:
            import keras
            from keras.utils import custom_object_scope
            custom_objects = {}
            lname = name.lower()
            if "resnet" in lname:
                from keras.applications.resnet50 import preprocess_input
                custom_objects["preprocess_input"] = preprocess_input
            if "inceptionv3" in lname:
                from keras.applications.inception_v3 import preprocess_input
                custom_objects["preprocess_input"] = preprocess_input
            if "mobilenetv2" in lname:
                from keras.applications.mobilenet_v2 import preprocess_input
                custom_objects["preprocess_input"] = preprocess_input
            with custom_object_scope(custom_objects):
                model = keras.models.load_model(path, compile=False)
            return model
        except Exception as e:
            st.error(f"Failed loading Keras model {name}: {e}")
            return None

    if fw == "torch_custom":
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 1)
            state = torch.load(path, map_location="cpu")
            sd = state.get("model", state) if isinstance(state, dict) else state
            model.load_state_dict(sd, strict=False)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Failed loading custom Torch model {name}: {e}")
            return None

    if fw == "torch":
        try:
            import torch
            model = torch.load(path, map_location="cpu")
            model.eval()
            return model
        except Exception as e:
            st.error(f"Failed loading Torch model {name}: {e}")
            return None

    if fw == "yolo":
        if not _ULTRA_AVAILABLE:
            st.error("Ultralytics YOLO not available.")
            return None
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Failed loading YOLO model {name}: {e}")
            return None

    return None

def preprocess_image(img: Image.Image, size: int):
    return np.asarray(img.resize((size, size))).astype(np.float32) / 255.0

def classify(model, arr: np.ndarray):
    x = np.expand_dims(arr, axis=0)
    try:
        import keras
        if isinstance(model, keras.Model):
            return model.predict(x)
    except Exception:
        pass
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            t = torch.tensor(x).permute(0,3,1,2).float()
            with torch.no_grad():
                out = model(t)
            return out.cpu().numpy()
    except Exception:
        pass
    return None

def interpret_class(preds):
    if preds is None:
        return None, None
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == 2:
        exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        idx = int(np.argmax(probs, axis=1)[0])
        return ["Male", "Female"][idx], float(probs[0][idx])
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0][0])
        prob = 1/(1+np.exp(-val))
        label = "Female" if prob >= 0.5 else "Male"
        return label, prob if label == "Female" else 1 - prob
    return None, None

def detect(model, img: Image.Image):
    arr = np.array(img.convert("RGB"))
    results = model.predict(source=arr)
    dets = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            try:
                coords = tuple(map(int, b.xyxy[0].cpu().numpy()))
            except:
                coords = tuple(map(int, b.xyxy[0]))
            name = model.names.get(cls, str(cls)) if hasattr(model, "names") else str(cls)
            dets.append((name, conf, coords))
    return dets

class GenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.info = MODELS_INFO[model_name]
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if self.info["type"] == "classification":
            arr = preprocess_image(pil, self.info["input_size"])
            preds = classify(self.model, arr)
            label, conf = interpret_class(preds)
            if label:
                draw.text((10, 10), f"{label} ({conf:.0%})", fill="red")
        else:
            for name, conf, box in detect(self.model, pil):
                draw.rectangle(box, outline="green", width=2)
                draw.text((box[0], max(box[1]-10,0)), f"{name} {conf:.2f}", fill="green")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

# Select model
safe_names = {re.sub(r"[^\w.-]", "_", n): n for n in MODELS_INFO}
choice = st.selectbox("Select model", list(safe_names.keys())) if MODELS_INFO else None
model_name = safe_names.get(choice) if choice else None
model = load_model(model_name, MODELS_INFO[model_name]) if model_name else None

# Upload image
st.subheader("\ud83d\udcf7 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if img_file and model:
    pil = Image.open(img_file).convert("RGB")
    st.image(pil, use_column_width=True)
    info = MODELS_INFO[model_name]
    if info["type"] == "classification":
        arr = preprocess_image(pil, info["input_size"])
        preds = classify(model, arr)
        label, conf = interpret_class(preds)
        if label:
            st.success(f"Prediction: {label} ({conf:.1%})")
    else:
        disp = pil.copy()
        draw = ImageDraw.Draw(disp)
        counts = {"male":0, "female":0}
        for name, conf, box in detect(model, pil):
            draw.rectangle(box, outline="green", width=2)
            draw.text((box[0], max(box[1]-10,0)), f"{name} {conf:.2f}", fill="green")
            lname = name.lower()
            if lname in counts: counts[lname] += 1
        st.image(disp, use_column_width=True)
        st.info(f"Detected: {counts.get('male',0)} males, {counts.get('female',0)} females")

# Live camera
st.subheader("\ud83d\udcf8 Live Camera Detection")
if model:
    webrtc_streamer(
        key="live-gender-detect",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=GenderProcessor,
        async_processing=True,
    )
else:
    st.warning("Please select a model first.")

st.markdown("---")
st.write(f"- Models from: {HF_REPO_ID}")
