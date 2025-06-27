import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import re
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.title("Drosophila Gender Detection")
st.write("Select mode and upload an image or use live camera.")

HF_REPO_ID = "RishiPTrial/models_h5"

# YOLO check
def check_ultralytics():
    try:
        import ultralytics
        version = ultralytics.__version__ if hasattr(ultralytics, "__version__") else "unknown"
        st.info(f"Ultralytics installed, version: {version}")
        return True
    except Exception as e:
        st.warning(f"Ultralytics import failed: {e}")
        return False

_ULTRA_AVAILABLE = check_ultralytics()

@st.cache_data(show_spinner=False)
def list_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith((".h5", ".pt", ".pth")) and not f.startswith(".")]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def build_model_info():
    info = {}
    for f in list_models():
        name_lower = f.lower()
        if name_lower.endswith(".pt"):
            info[f] = {"type": "detection", "framework": "yolo"}
        elif f == "model_final.pth":
            info[f] = {"type": "classification", "framework": "torch_custom", "input_size": 224}
        elif name_lower.endswith(".pth"):
            info[f] = {"type": "classification", "framework": "torch", "input_size": 224}
        elif name_lower.endswith(".h5"):
            size = 299 if "inceptionv3" in name_lower else 224
            info[f] = {"type": "classification", "framework": "keras", "input_size": size}
    return info

MODELS_INFO = build_model_info()

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
                return keras.models.load_model(path, compile=False)
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
            from ultralytics import YOLO
            return YOLO(path)
        except Exception as e:
            st.error(f"Failed loading YOLO model {name}: {e}")
            return None

    return None

def preprocess_image(img, size):
    return np.asarray(img.resize((size, size))).astype(np.float32) / 255.0

def classify(model, arr):
    x = np.expand_dims(arr, axis=0)
    try:
        import keras
        if isinstance(model, keras.Model):
            return model.predict(x)
    except: pass
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            t = torch.tensor(x).permute(0,3,1,2).float()
            with torch.no_grad():
                return model(t).cpu().numpy()
    except: pass
    return None

def interpret(preds):
    if preds is None: return None, None
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == 2:
        probs = np.exp(arr - np.max(arr))
        probs = probs / probs.sum(axis=1, keepdims=True)
        idx = int(np.argmax(probs, axis=1)[0])
        return ["Male", "Female"][idx], float(probs[0][idx])
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0][0])
        prob = 1/(1+np.exp(-val))
        label = "Female" if prob >= 0.5 else "Male"
        return label, prob if label=="Female" else 1-prob
    return None, None

def detect(model, img):
    arr = np.array(img.convert("RGB"))
    results = model.predict(source=arr)
    dets = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            coords = tuple(map(int, b.xyxy[0].cpu().numpy()))
            name = model.names.get(cls, str(cls))
            dets.append((name, conf, coords))
    return dets

mode = st.radio("Choose mode", ["Single", "Ensemble (Classifiers Only)"])

if mode == "Single":
    safe_names = {re.sub(r"[^\w.-]", "_", n): n for n in MODELS_INFO}
    choice = st.selectbox("Select model", list(safe_names.keys()))
    model_name = safe_names.get(choice)
    model = load_model(model_name, MODELS_INFO[model_name]) if model_name else None
    
    st.subheader("Upload Image")
    img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if img_file and model:
        pil = Image.open(img_file).convert("RGB")
        st.image(pil, use_column_width=True)
        info = MODELS_INFO[model_name]
        if info["type"] == "classification":
            arr = preprocess_image(pil, info["input_size"])
            label, conf = interpret(classify(model, arr))
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
else:
    chosen = st.multiselect("Pick models for ensemble", [n for n,i in MODELS_INFO.items() if i['type']=="classification"])
    if chosen:
        models = [(n, load_model(n, MODELS_INFO[n])) for n in chosen]
        st.subheader("Upload Image")
        img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
        if img_file:
            pil = Image.open(img_file).convert("RGB")
            st.image(pil, use_column_width=True)
            votes = []
            confs = {"Male": [], "Female": []}
            for name, mdl in models:
                size = MODELS_INFO[name]["input_size"]
                arr = preprocess_image(pil, size)
                label, conf = interpret(classify(mdl, arr))
                if label:
                    votes.append(label)
                    confs[label].append(conf)
            if votes:
                from collections import Counter
                count = Counter(votes)
                most_common = count.most_common()
                if len(most_common) == 1 or most_common[0][1] != most_common[1][1]:
                    result = most_common[0][0]
                else:
                    avg_conf = {k: np.mean(confs[k]) for k in confs if confs[k]}
                    result = max(avg_conf.items(), key=lambda x: x[1])[0]
                st.success(f"Ensemble prediction: {result}")

st.markdown("---")
st.write(f"- Models from: {HF_REPO_ID}")
