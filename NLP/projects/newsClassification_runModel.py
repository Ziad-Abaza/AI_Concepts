import numpy as np
import re
from tensorflow.keras.models import load_model  
import joblib
import spacy

# تحميل النموذج ومعالج النصوص والبيانات
nlp = spacy.load("en_core_web_md")
model = load_model('news_classifier_model.h5')

vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# تنظيف النصوص
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# معالجة النصوص باستخدام Spacy
def preprocess_with_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# استخراج الكيانات
def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return ' '.join(entities)

# دمج النصوص مع الكيانات وتوقع الفئات
def predict_category(texts):
    cleaned_texts = [clean_text(text) for text in texts]
    preprocess_texts = [preprocess_with_spacy(text) for text in cleaned_texts]
    
    # استخراج الكيانات لكل نص ودمجها مع النصوص المعالجة
    texts_with_entities = [preprocess_text + ' ' + extract_entities(preprocess_text) for preprocess_text in preprocess_texts]
    
    # تحويل النصوص إلى تمثيلات عددية باستخدام المتجهات المحفوظة
    x_new = vectorizer.transform(texts_with_entities)
    
    # التنبؤ بالفئات
    y_pred_new = model.predict(x_new)
    y_pred_classes_new = np.argmax(y_pred_new, axis=1)
    
    # تحويل الفئات إلى الأسماء الأصلية باستخدام الترميز
    return [label_encoder.classes_[i] for i in y_pred_classes_new]

# اختبار النصوص الجديدة
new_texts = [
    "find WhatsApp's updated privacy policy unacceptable: Inshorts poll",
    "Elon Musk tweets 'as promised' regarding Tesla's entry into India",
    "Congress MLA says BJP without Raje and Congress without Gehlot can't be imagined",
    "The new Volvo S60 is designed to save lives and offer an exhilarating driving experience",
    "Tata Motors' tweet sparks rumors of a tie-up with Tesla, but the company denies it"
]

# تنبؤ الفئات لكل نص
predicted_categories = predict_category(new_texts)

# طباعة النتائج
for text, category in zip(new_texts, predicted_categories):
    print(f"Text: {text}\nPredicted Category: {category}\n")
