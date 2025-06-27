import sys
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import re
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from collections import Counter

# ————————— Streamlit Page Setup —————————
st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.title("Drosophila Gender Detection")
st.write("Select a model (or ensemble) and upload an image or use live camera.")

HF_REPO_ID = "RishiPTrial/models_h5"

# ————————— Check YOLO Availability —————————
def check_ultralytics():
    try:
        import ultralytics
        ver = ultralytics.__version__ if hasattr(ultralytics, "__version__") else "unknown"
        st.info(f"Ultralytics installed, version: {ver}")
        return True
    except Exception as e:
        st.warning(f"Ultralytics import failed: {e}")
        return False
_ULTRA_AVAILABLE = check_ultralytics()

# ————————— List & Catalog Models —————————
@st.cache_data(show_spinner=False)
def list_hf_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith((".pt",".h5",".pth")) and not f.startswith(".")]
    except:
        return []

@st.cache_data(show_spinner=False)
def build_models_info():
    info = {}
    for fname in list_hf_models():
        lower = fname.lower()
        size = 299 if "inceptionv3" in lower else 224
        if lower.endswith(".pt"):
            info[fname] = {"type":"detection","framework":"yolo"}
        elif lower.endswith(".h5"):
            info[fname] = {"type":"classification","framework":"keras","input_size":size}
        elif fname == "model_final.pth":
            info[fname] = {"type":"classification","framework":"torch_custom","input_size":size}
        elif lower.endswith(".pth"):
            info[fname] = {"type":"classification","framework":"torch","input_size":size}
    return info

MODELS_INFO = build_models_info()
if not MODELS_INFO:
    st.error(f"No models found in {HF_REPO_ID}")

# ————————— Model Loading Helpers —————————
def load_model_final_pth(path):
    import torch, torch.nn as nn
    from torchvision import models
    m = models.resnet18(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features,1)
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model", ckpt) if isinstance(ckpt,dict) else ckpt
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

@st.cache_resource(show_spinner=False)
def load_model_from_hf(name, info):
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    except Exception as e:
        st.error(f"Download error {name}: {e}")
        return None

    fw = info["framework"]
    if fw == "keras":
        try:
            import tensorflow as tf
            from tensorflow.keras.utils import custom_object_scope
            custom_objs = {}
            lname = name.lower()
            if "resnet50" in lname:
                from tensorflow.keras.applications.resnet50 import preprocess_input
                custom_objs["preprocess_input"] = preprocess_input
            elif "inceptionv3" in lname:
                from tensorflow.keras.applications.inception_v3 import preprocess_input
                custom_objs["preprocess_input"] = preprocess_input
            elif "mobilenetv2" in lname:
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                custom_objs["preprocess_input"] = preprocess_input
            with custom_object_scope(custom_objs):
                return tf.keras.models.load_model(path, compile=False)
        except ImportError:
            st.error("TensorFlow not available for Keras models.")
            return None
        except Exception as e:
            st.error(f"Failed loading Keras model {name}: {e}")
            return None

    if fw == "torch_custom":
        try:
            return load_model_final_pth(path)
        except Exception as e:
            st.error(f"Failed loading custom PyTorch model {name}: {e}")
            return None

    if fw == "torch":
        try:
            import torch
            m = torch.load(path, map_location="cpu")
            m.eval()
            return m
        except Exception as e:
            st.error(f"Failed loading PyTorch model {name}: {e}")
            return None

    if fw == "yolo":
        if not _ULTRA_AVAILABLE:
            st.error("YOLO not installed; cannot load detection model.")
            return None
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            if not hasattr(model,'names'):
                model.names = {0:"Male",1:"Female"}
            return model
        except Exception as e:
            st.error(f"Failed loading YOLO model {name}: {e}")
            return None

    st.error(f"Unsupported framework for {name}")
    return None

# ————————— Preprocessing & Inference —————————
def preprocess_pil(img: Image.Image, size:int):
    arr = img.resize((size,size))
    return np.asarray(arr).astype(np.float32)/255.0

def classify(model, arr:np.ndarray):
    x = np.expand_dims(arr,0)
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return model.predict(x)
    except: pass
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                t = torch.tensor(x).permute(0,3,1,2).float()
                return model(t).cpu().numpy()
    except: pass
    return None

def interpret_classification(preds):
    if preds is None: return None,0.0
    p = np.asarray(preds).flatten()
    if p.size==2:
        probs = np.exp(p)/np.sum(np.exp(p))
        idx = int(np.argmax(probs))
        return ["Male","Female"][idx], float(probs[idx])
    if p.size==1:
        prob = 1/(1+np.exp(-p[0]))
        lbl = "Female" if prob>=0.5 else "Male"
        return lbl, prob if lbl=="Female" else 1-prob
    return None,0.0

def detect_yolo(model, pil:Image.Image):
    try:
        arr = np.array(pil.convert("RGB"))
        res = model.predict(source=arr, verbose=False)[0]
    except:
        return []
    dets=[]
    for b in res.boxes:
        cls = int(b.cls[0]); conf=float(b.conf[0])
        coords = list(map(int,b.xyxy[0].cpu().numpy()))
        dets.append({"label":model.names.get(cls,str(cls)),"conf":conf,"box":coords})
    return dets

# ————————— UI: Model / Ensemble Selection —————————
st.sidebar.header("Mode Selection")
mode = st.sidebar.radio("Mode", ["Single Model","Ensemble (Classifiers)"])
if mode=="Single Model":
    choice = st.sidebar.selectbox("Select model", list(MODELS_INFO.keys()))
    MODELS = [(choice, load_model_from_hf(choice, MODELS_INFO[choice]))]
else:
    cls_list = [n for n,i in MODELS_INFO.items() if i["type"]=="classification"]
    chosen = st.sidebar.multiselect("Pick classifiers", cls_list, default=cls_list)
    MODELS = [(n, load_model_from_hf(n, MODELS_INFO[n])) for n in chosen]

# ————————— UI: Image Upload & Prediction —————————
st.markdown("---")
st.subheader("📷 Upload Image")
img_file = st.file_uploader("", type=["jpg","jpeg","png"])
if img_file:
    pil = Image.open(img_file).convert("RGB")
    st.image(pil, use_column_width=True)
    votes=[]; confs={}
    draw = ImageDraw.Draw(pil)

    for name, mdl in MODELS:
        info = MODELS_INFO[name]
        if info["type"]=="classification":
            arr = preprocess_pil(pil, info["input_size"])
            lbl, c = interpret_classification(classify(mdl,arr))
            votes.append(lbl); confs.setdefault(lbl,[]).append(c)
        else:
            for d in detect_yolo(mdl,pil):
                votes.append(d["label"])
                confs.setdefault(d["label"],[]).append(d["conf"])
                x1,y1,x2,y2 = d["box"]
                draw.rectangle([x1,y1,x2,y2],outline="green",width=2)
                draw.text((x1,y1-12),f'{d["label"]} {d["conf"]:.2f}',fill="green")

    if votes:
        cnt = Counter(votes).most_common()
        if mode.startswith("Ensemble") and len(cnt)>1 and cnt[0][1]==cnt[1][1]:
            avg = {l:np.mean(confs[l]) for l in confs}
            final = max(avg,key=avg.get)
            final_conf = avg[final]
        else:
            final = cnt[0][0]
            # if single model, grab that one confidence; if ensemble, use its mean
            final_conf = confs[final][0] if len(MODELS)==1 else np.mean(confs[final])
        st.success(f"Final Prediction: {final} ({final_conf:.1%})")

# ————————— UI: Live Camera —————————
st.markdown("---")
st.subheader("📸 Live Camera")
if MODELS:
    def factory():
        class P(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="rgb24")
                pil = Image.fromarray(img)
                draw = ImageDraw.Draw(pil)
                votes=[]; confs={}
                for name,mdl in MODELS:
                    info = MODELS_INFO[name]
                    if info["type"]=="classification":
                        arr = preprocess_pil(pil, info["input_size"])
                        lbl,c = interpret_classification(classify(mdl,arr))
                        votes.append(lbl); confs.setdefault(lbl,[]).append(c)
                    else:
                        for d in detect_yolo(mdl,pil):
                            votes.append(d["label"])
                            confs.setdefault(d["label"],[]).append(d["conf"])
                            x1,y1,x2,y2 = d["box"]
                            draw.rectangle([x1,y1,x2,y2],outline="green",width=2)
                            draw.text((x1,y1-12),f'{d["label"]} {d["conf"]:.2f}',fill="green")
                if votes:
                    cnt = Counter(votes).most_common()
                    if mode.startswith("Ensemble") and len(cnt)>1 and cnt[0][1]==cnt[1][1]:
                        avg = {l:np.mean(confs[l]) for l in confs}
                        final = max(avg,key=avg.get)
                        final_conf = avg[final]
                    else:
                        final = cnt[0][0]
                        final_conf = confs[final][0] if len(MODELS)==1 else np.mean(confs[final])
                    draw.text((10,10), f"{final} ({final_conf:.1%})", fill="red")
                return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")
        return P()

    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video":True,"audio":False},
        video_processor_factory=factory,
        async_processing=True
    )
else:
    st.warning("No models loaded.")

# ————————— Footer Notes —————————
st.markdown("---")
st.write("**Notes:** Models hosted on Hugging Face:", HF_REPO_ID)
