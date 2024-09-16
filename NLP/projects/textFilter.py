from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob
import pandas as pd
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()

# تحميل كلمات التوقف باللغة الإنجليزية
stop_words = set(stopwords.words('english'))

df = pd.read_json('assets/textFilter.json')

# استخراج البيانات من الحقل 'data' في JSON
dataFrame = df['data']
textData = []

###############################################################
# دالة لتنظيف النصوص من العلامات غير المرغوبة
# تقوم بإزالة أي علامات غير حروف أو مسافات
###############################################################
def cleanText(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

###############################################################
# معالجة كل نص في البيانات
# - تحويل النص إلى حروف صغيرة
# - تنظيف النص من العلامات
# - تقسيم النص إلى كلمات باستخدام word_tokenize
# - تصفية الكلمات باستخدام Lemmatizer وإزالة كلمات التوقف
###############################################################
for data in dataFrame:
    item = data['text'].lower()  # تحويل النص إلى حروف صغيرة
    item = cleanText(item)  # تنظيف النص من العلامات غير الضرورية
    words = nltk.word_tokenize(item)  # تقسيم النص إلى كلمات
    # تصفية الكلمات بإزالة كلمات التوقف وتطبيق عملية التحجيم
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # جمع الكلمات المفلترة في نص واحد
    filtered_text = ' '.join(filtered_words)
    # تصحيح الأخطاء الإملائية
    corrected_text = TextBlob(filtered_text).correct()
    # إضافة النص المصحح إلى القائمة
    textData.append(str(corrected_text))  # تحويل إلى string للتأكد من النوع

print("---------------------------\ SHOW DATA AFTER FILTERING /---------------------------")
print('Data After Filtering: ', textData)

###############################################################
# إستخراج أرقام الهواتف - الإميلات - الروابط من النصوص 
###############################################################
print("---------------------------\ EXTRACT INFORMATION /---------------------------")
def extract_info(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    emails = re.findall(r'\S+@\S+', text)    
    phones = re.findall(r'\+?\d[\d\s-]{8,}\d', text)
    return urls, emails, phones

for data in dataFrame:
    urls, emails, phones = extract_info(data['text'])
    print(f"--------------\ -{data['id']}- /--------------")
    print(f"Text: {data['text']}")
    print(f"URLs: {urls}")
    print(f"Emails: {emails}")
    print(f"Phones: {phones}")

###############################################################
# تطبيق CountVectorizer على النصوص المحفوظة في textData
# وتحويلها إلى تمثيل رقمي (Bag of Words)
###############################################################
X = vectorizer.fit_transform(textData)

###############################################################
# استخراج أسماء الكلمات الفريدة من CountVectorizer
# هذه هي الأعمدة التي تمثل الكلمات في التمثيل الرقمي
###############################################################
feature_names = vectorizer.get_feature_names_out()

###############################################################
# إنشاء DataFrame باستخدام Pandas لعرض تمثيل Bag of Words
# - الأعمدة تمثل الكلمات الفريدة
# - الصفوف تمثل النصوص الأصلية بعد تحويلها إلى أرقام
###############################################################
table_df = pd.DataFrame(X.toarray(), columns=feature_names)

print("---------------------------\ BAG OF WORDS /---------------------------")
print(table_df)
