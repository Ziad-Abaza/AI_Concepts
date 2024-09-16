from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from nltk.corpus import stopwords
import pandas as pd
import joblib
import nltk
import re
import spacy

nlp = spacy.load("en_core_web_md")
stop_words = set(stopwords.words('english'))

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv('assets/news/inshort_news_data-2.csv')
df = df[['news_article', 'news_category']]  # اختيار المقالات والتصنيفات
df.columns = ['article', 'category']  # إعادة تسمية الأعمدة لأسماء أسهل في التعامل

###############################################################
# دالة لتنظيف النصوص من الأحرف غير المرغوبة وتحويلها إلى أحرف صغيرة
###############################################################
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # إزالة الأحرف غير الأبجدية
    text = re.sub(r'\s+', ' ', text).strip()  # إزالة المسافات الزائدة
    text = text.lower()  # تحويل النص إلى أحرف صغيرة
    return text

###############################################################
# دالة لاستخدام Spacy لمعالجة النصوص، إزالة الكلمات الشائعة، واستخراج الجذور
################################################################
def preprocess_with_spacy(text):
    doc = nlp(text)  
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]  
    return ' '.join(tokens)

###############################################################
# دالة لاستخراج الكيانات المسماة من النص باستخدام Spacy 
################################################################

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]  # استخراج الكيانات
    return ' '.join(entities)

###############################################################
# تحويل النصوص إلى كلمات متقطعة وإزالة الكلمات الشائعة باستخدام NLTK
###############################################################
tokens = [[word for word in nltk.word_tokenize(article) if word.lower() not in stop_words] for article in df['article']]
articles = [' '.join(token_list) for token_list in tokens]  # دمج الكلمات في نصوص جديدة بعد إزالة الكلمات الشائعة

###############################################################
# تطبيق الدوال لتنظيف النصوص واستخراج الكيانات
###############################################################
df['article'] = df['article'].apply(clean_text)  # تنظيف النصوص
df['article'] = df['article'].apply(preprocess_with_spacy)  # معالجة النصوص باستخدام Spacy
df['entities'] = df['article'].apply(extract_entities)  # استخراج الكيانات من النصوص

# دمج النصوص المعالجة مع الكيانات في عمود المقالة
df['article'] = df['article'] + ' ' + df['entities']

###############################################################
# تحويل النصوص إلى تمثيلات عددية باستخدام TF-IDF
###############################################################
vectorizer = TfidfVectorizer()  # إنشاء كائن TF-IDF
x = vectorizer.fit_transform(articles)  # تحويل النصوص إلى تمثيلات عددية

# ترميز الفئات باستخدام LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])  # ترميز التصنيفات
y = to_categorical(y)  # تحويل التصنيفات إلى One-Hot Encoding

###############################################################
# إنشاء النموذج العصبي
###############################################################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # تقسيم البيانات إلى تدريب واختبار

# بناء النموذج باستخدام تسلسل طبقات Dense و Dropout لتجنب الإفراط في التعلم
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),  # طبقة مخفية
    Dropout(0.5),  # إسقاط 50% من الخلايا بشكل عشوائي لتجنب الإفراط في التعلم
    Dense(64, activation='relu'),  # طبقة مخفية أخرى
    Dropout(0.5),  # إسقاط 50% أخرى
    Dense(y_train.shape[1], activation='softmax')  # الطبقة النهائية مع تفعيل softmax للتصنيف المتعدد
])

# تجميع النموذج باستخدام المحسن Adam وخسارة التصنيف التعددي
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

###############################################################
# إضافة EarlyStopping لإيقاف التدريب تلقائيًا إذا لم يتحسن الأداء
###############################################################
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # إيقاف التدريب إذا لم يتحسن الأداء بعد 5 محاولات

# تدريب النموذج مع استخدام EarlyStopping
history = model.fit(x_train, y_train, epochs=70, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

###############################################################
# حفظ النموذج والمعالجات المستخدمة
###############################################################
# model.save('news_classifier_model.h5')  # حفظ النموذج
# joblib.dump(vectorizer, 'vectorizer.pkl')  # حفظ كائن TF-IDF
# joblib.dump(label_encoder, 'label_encoder.pkl')  # حفظ كائن ترميز التصنيفات

###############################################################
# تقييم النموذج على مجموعة الاختبار
###############################################################
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss* 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%") 
