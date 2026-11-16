from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.pipeline import LeafCNN

st.set_page_config(page_title="PlantGuard AI", page_icon="🌿", layout="centered")
st.title("🌿 PlantGuard AI")
st.write("Domates yaprağı görüntüsünden hastalık sınıflandırma — eğitim amaçlı prototip")
model_path = Path(__file__).parent / "outputs" / "model.pt"
if not model_path.exists():
    st.warning("Önce veri setini yerleştirip `python -m src.pipeline` komutuyla modeli eğitin.")
    st.stop()
@st.cache_resource
def load_model():
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model = LeafCNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    return model, ckpt["classes"], ckpt["image_size"]
model, classes, size = load_model()
file = st.file_uploader("Yaprak fotoğrafı yükleyin", type=["jpg","jpeg","png"])
if file:
    image = Image.open(file).convert("RGB"); st.image(image, caption="Yüklenen görüntü", use_container_width=True)
    transform = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)])
    with torch.no_grad():
        prob = torch.softmax(model(transform(image).unsqueeze(0)),dim=1)[0]
    idx = int(prob.argmax())
    st.subheader(f"Tahmin: {classes[idx]}")
    st.metric("Model skoru (kalibre edilmiş güven değil)", f"%{prob[idx].item()*100:.1f}")
    st.bar_chart({classes[i]:float(prob[i]) for i in range(len(classes))})
st.caption("Bu uygulama laboratuvar görüntüleriyle eğitilebilir; tarımsal teşhis veya ilaçlama kararı için tek başına kullanılmamalıdır.")
