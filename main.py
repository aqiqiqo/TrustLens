import streamlit as st
from transformers import pipeline, Pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import time
from PIL import Image
import imagehash
import io
import math
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch


MODEL_CV = "prithivMLmods/Deep-Fake-Detector-v2-Model"


image_processor = AutoImageProcessor.from_pretrained(MODEL_CV)
model_cv = AutoModelForImageClassification.from_pretrained(MODEL_CV)

def analyze_image(image_path):
    image = Image.open(image_path)
    inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_cv(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = probs.max().item()
        label = model_cv.config.id2label[probs.argmax().item()]
    return {"label": label, "confidence": confidence}

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

MODEL_NAME =  "sereotubu/fake-news-detector-isot"
REPORTS_DIR = "reports"
IMAGE_DB_DIR = "image_db"  

SENSATIONAL_WORDS = {
    "shocking","unbelievable","breaking","exclusive","must-see","miracle",
    "secret","you won't believe","this will change","revealed","exposed"
}


W_MODEL = 0.5
W_TEXTHEUR = 0.4
W_MEDIA = 0.1



@st.cache_resource(show_spinner=False)
def load_model():
    """Load transformers pipeline once."""
    try:
        clf = pipeline("text-classification", model=MODEL_NAME)
        return clf
    except Exception as e:
        st.error("Не удалось загрузить модель через pipeline. Проверьте соединение и зависимости.")
        st.write(e)
        return None

def text_heuristics(text: str):
    """Return simple heuristic scores in 0..1 for manipulative language."""
    if not text:
        return {
            "sensational_ratio": 0.0,
            "caps_ratio": 0.0,
            "exclaim_ratio": 0.0,
            "length_norm": 0.0
        }
    txt_lower = text.lower()
    words = text.split()
    n_words = max(len(words), 1)


    sensational_count = sum(1 for w in SENSATIONAL_WORDS if w in txt_lower)
    sensational_ratio = min(1.0, sensational_count / 3.0)


    caps_words = sum(1 for w in words if sum(1 for c in w if c.isupper()) > 0 and (sum(1 for c in w if c.isupper())/max(len(w),1) > 0.6))
    caps_ratio = min(1.0, caps_words / max(1, n_words))


    exclaim_count = text.count("!")
    exclaim_ratio = min(1.0, exclaim_count / 3.0)


    length_norm = 1.0 - min(1.0, math.log(n_words + 1) / math.log(200 + 1)) 

    return {
        "sensational_ratio": sensational_ratio,
        "caps_ratio": caps_ratio,
        "exclaim_ratio": exclaim_ratio,
        "length_norm": length_norm
    }

def compute_text_score(heur: dict):
    

    score = (0.4 * heur["sensational_ratio"] +
             0.25 * heur["caps_ratio"] +
             0.2 * heur["exclaim_ratio"] +
             0.15 * heur["length_norm"])
    return min(1.0, max(0.0, score))

def compute_trust_score(model_label: str, model_score: float, text_manip_score: float, media_flag: float):
   

    if model_label.upper() == "FAKE":
        model_trust = 1 - model_score
    else:
        model_trust = model_score

    
    text_trust = 1 - text_manip_score

    
    media_trust = 1 - media_flag

    combined = W_MODEL * model_trust + W_TEXTHEUR * text_trust + W_MEDIA * media_trust
    trust_percent = int(round(100 * combined))
    trust_percent = max(0, min(100, trust_percent))
    return trust_percent

def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(IMAGE_DB_DIR, exist_ok=True)

def save_report(report: dict):
    ensure_dirs()
    ts = int(time.time())
    fname = os.path.join(REPORTS_DIR, f"report_{ts}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return fname

def image_similarity_hash(pil_img: Image.Image):
    """Compute perceptual hash (pHash) for the image and compare with IMAGE_DB_DIR files."""
    h = imagehash.phash(pil_img)
    similar = []
    if not os.path.exists(IMAGE_DB_DIR):
        return {"hash": str(h), "matches": []}
    for fn in os.listdir(IMAGE_DB_DIR):
        fp = os.path.join(IMAGE_DB_DIR, fn)
        try:
            db_img = Image.open(fp)
            db_h = imagehash.phash(db_img)
            dist = h - db_h 
            if dist <= 8:
                similar.append({"file": fn, "distance": int(dist)})
        except Exception:
            continue
    return {"hash": str(h), "matches": similar}


def main():
    st.set_page_config(page_title="MVP Fake & Deepfake Detector", page_icon="🕵️‍♀️", layout="centered")
    st.title(" Детектор фейковых новостей ")



    
    col1, col2 = st.columns([3,1])

    with col1:
        
        mode = st.radio("Режим проверки", ["Текст", "Изображение", "Оба"], index=0)

        user_text = ""
        uploaded_image = None

        if mode in ("Текст", "Оба"):
            user_text = st.text_area("Вставьте текст новости или заголовок", height=200)
        if mode in ("Изображение", "Оба"):
            uploaded_file = st.file_uploader("Загрузите изображение (jpg/png)", type=["jpg","jpeg","png"])
            if uploaded_file is not None:
                uploaded_image = Image.open(io.BytesIO(uploaded_file.read()))
                st.image(uploaded_image, caption="Загруженное изображение")

        run = st.button("Проверить")

    

    if run:
        ensure_dirs()
        t0 = time.time()
        
        with st.spinner("Загружаем модель (если ещё не загружена)..."):
            clf = load_model()
        
        if clf is None:
            st.error("Модель недоступна — используем заглушку (assume REAL с низкой уверенностью).")
            model_label = "REAL"
            model_conf = 0.6
        else:
            if mode in ("Текст", "Оба") and user_text.strip():
                with st.spinner("Анализируем текст..."):
                    try:
                        res = clf(user_text[:1000])  
                        if isinstance(res, list):
                            r0 = res[0]
                        else:
                            r0 = res
                        model_label = r0.get("label", "REAL")
                        model_conf = float(r0.get("score", 0.5))
                    except Exception as e:
                        st.write("Ошибка при вызове модели:", e)
                        model_label = "REAL"
                        model_conf = 0.5
            else:
                
                model_label = "REAL"
                model_conf = 0.5

        
        heur = text_heuristics(user_text)
        text_manip_score = compute_text_score(heur)

        
        media_flag = 0.0
        image_info = None
        if uploaded_image is not None:
            with st.spinner("Анализируем изображение (pHash сравнение)..."):
                image_info = image_similarity_hash(uploaded_image)
                matches = image_info.get("matches", [])
                if matches:
                    
                    media_flag = min(1.0, 0.6 + 0.1 * len(matches))
                else:
                    media_flag = 0.0

        trust = compute_trust_score(model_label, model_conf, text_manip_score, media_flag)

        
        st.header("Результат анализа")
        st.subheader(f"Итоговая оценка доверия: {trust}%")
        st.progress(trust / 100)

        st.markdown("### Модельная оценка (NLP)")

# Перевод меток модели
        label_map = {
            "NEGATIVE": "Ложная информация",
            "FAKE": "Ложная информация",
            "POSITIVE": "Достоверно",
            "REAL": "Достоверно",
            "Fake": "Ложная информация",
            "Real": "Достоверно"
        }
        user_friendly_label = label_map.get(model_label, model_label)

        st.write(f"- Метка модели: **{user_friendly_label}**")
        st.write(f"- Уверенность модели: **{model_conf:.2f}**")


        st.markdown("### Текстовые признаки манипуляции (эвристики)")
        st.write(f"- Сенсационные слова (fraction): **{heur['sensational_ratio']:.2f}**")
        st.write(f"- Соотношение CAPS-слов: **{heur['caps_ratio']:.2f}**")
        st.write(f"- Восклицательные знаки: **{heur['exclaim_ratio']:.2f}**")
        st.write(f"- Короткость/Headline фактор: **{heur['length_norm']:.2f}**")
        st.write(f"- Итоговый текстовый манипулятивный скор: **{text_manip_score:.2f}**")

        st.markdown("### Анализ изображения (pHash)")
        if image_info is not None:
            st.write(f"- pHash изображения: `{image_info.get('hash')}`")
            if image_info.get("matches"):
                st.write("- Найдены похожие изображения в local image_db (возможная переиспользованная картинка):")
                for m in image_info["matches"]:
                    st.write(f"  - {m['file']} (hamming distance = {m['distance']})")
                st.warning("Найденные совпадения повышают подозрительность")
            else:
                st.write("- Совпадений не найдено в локальной базе (image_db).")
        else:
            st.write("- Изображение не загружено.")

        
        
        for r in reasons:
            st.write("- " + r)

        
        report = {
            "timestamp": int(t0),
            "input_text": user_text,
            "model_label": model_label,
            "model_confidence": model_conf,
            "heuristics": heur,
            "text_manip_score": text_manip_score,
            "image_info": image_info,
            "media_flag": media_flag,
            "trust": trust
        }
        saved = save_report(report)
        st.caption(f"Отчёт сохранён: {saved}")

        st.success(f"Готово — анализ завершён за {int(time.time()-t0)} сек.")

    
    st.markdown("---")
if __name__ == "__main__":
    main()

