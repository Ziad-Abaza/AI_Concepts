import spacy
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from nltk.corpus import stopwords
import nltk
import joblib

# تحميل موديل SpaCy
nlp = spacy.load("en_core_web_md")
stop_words = set(stopwords.words('english'))

# تحميل NLTK
nltk.download('stopwords')
nltk.download('punkt')

# تحميل البيانات
df = pd.read_csv('assets/news/inshort_news_data-2.csv')
df = df[['news_article', 'news_category']]
df.columns = ['article', 'category']

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def preprocess_with_spacy(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return ' '.join(entities)

# معالجة النصوص باستخدام SpaCy وإضافة الكيانات المستخرجة
df['article'] = df['article'].apply(clean_text)
df['cleaned_article'] = df['article'].apply(preprocess_with_spacy)
df['entities'] = df['article'].apply(extract_entities)

# دمج النصوص المنظفة والكيانات المستخرجة
df['processed_text'] = df['cleaned_article'] + ' ' + df['entities']

# تحويل النصوص إلى تمثيلات عددية
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['processed_text'])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])
y = to_categorical(y)  # تحويل التصنيفات إلى ترميز One-Hot

# إنشاء النموذج
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=70, batch_size=32, validation_split=0.2)

# model.save('news_classifier_model.h5')
# joblib.dump(vectorizer, 'vectorizer.pkl')
# joblib.dump(label_encoder, 'label_encoder.pkl')

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss* 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
