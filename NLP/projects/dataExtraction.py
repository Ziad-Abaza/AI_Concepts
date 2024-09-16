from sklearn.feature_extraction.text import CountVectorizer
from spellchecker import SpellChecker
from nltk.corpus import stopwords
import spacy
import nltk
import re


nlp = spacy.load("en_core_web_md")  
nltk.download('stopwords') 

spell = SpellChecker()
stop_words = set(stopwords.words('english'))  

###############################################
# تنظيف النصوص
# تقوم هذه الدالة بإزالة الرموز التعبيرية، الأحرف الخاصة غير المرغوب فيها،
# تصحيح الإملاء، وتحويل النص إلى حالة صغيرة. كما تقوم بإزالة الكلمات غير المهمة (stopwords).
###############################################
def clean_text(text):
    # Remove emojis
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    # Remove extra spaces and new lines
    text = re.sub(r'\s+', ' ', text).strip() 
    # Remove special characters, except @, . and :
    text = re.sub(r'[^\w\s@.:/-]', '', text)
    # Remove extra spaces around colons and slashes
    text = re.sub(r'\s*(:|/)\s*', r'\1', text) 
    # Transform to lower case
    text = text.lower() 
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    # Correct spelling
    words = text.split()
    corrected_words = [spell.candidates(word).pop() if spell.candidates(word) else word for word in words] 
    text = ' '.join(corrected_words) 
    return text

###############################################
# تصفية النصوص باستخدام POS tagging
# تقوم هذه الدالة باختيار الكلمات التي تحمل أجزاء الكلام (POS) ذات الأهمية فقط مثل الأسماء والأفعال والصفات.
###############################################
def data_classification(text):
    doc = nlp(text) 
    filtered_words = [token.text for token in doc if token.text.lower() not in stop_words and token.pos_ in {'NOUN', 'ADJ', 'VERB', 'ADV'}]
    return ' '.join(filtered_words) 

###############################################
# استخراج الكيانات المسماة (NER)
# تقوم هذه الدالة باستخراج الكيانات المسماة من النص مثل الأسماء والأماكن والتواريخ
# باستخدام نموذج spaCy.
###############################################
def chick_entities(text):
    doc = nlp(text)  
    entities = [(ent.text, ent.label_) for ent in doc.ents] 
    return entities 

###############################################
# تحليل bigram
# تقوم هذه الدالة بتحليل الbigram (الزوجيات) للنصوص النظيفة والمصفاة
# باستخدام CountVectorizer من مكتبة sklearn.
###############################################
def bigram_analysis(text):
    cleaned_text = clean_text(text)  # تنظيف النصوص
    filtered_text = data_classification(cleaned_text)  
    vectorizer = CountVectorizer(ngram_range=(2, 2)) 
    bigram_matrix = vectorizer.fit_transform([filtered_text])  
    return bigram_matrix.toarray(), vectorizer.get_feature_names_out()  

###############################################
# تحليل trigram
# تقوم هذه الدالة بتحليل الtrigram (الثلاثيات) للنصوص النظيفة والمصفاة
# باستخدام CountVectorizer من مكتبة sklearn.
###############################################
def trigram_analysis(text):
    cleaned_text = clean_text(text)  # تنظيف النصوص
    filtered_text = data_classification(cleaned_text) 
    vectorizer = CountVectorizer(ngram_range=(3, 3))  
    trigram_matrix = vectorizer.fit_transform([filtered_text])  
    return trigram_matrix.toarray(), vectorizer.get_feature_names_out()  


# Example usage
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "The sun was shining brightly in the clear blue sky.",
    "The cat purrs contentedly on my lap.",
    "Natural Language Processing (NLP) is a fascinating field of artificial intelligence that enables computers to understand, interpret, and generate human language. It combines computational linguistics with machine learning techniques to process large amounts of natural language data. Applications of NLP include speech recognition, sentiment analysis, and language translation.",
    "The company's mission is to provide innovative solutions to complex problems, leveraging cutting-edge technologies to  drive business growth and customer satisfaction.",
    "The new policy aims to reduce carbon emissions by 50% within the next five years,  promoting sustainable practices and environmentally friendly technologies,  and fostering a culture of innovation and collaboration.",
    "my name  is mohamed and i am a software engineer.  i have a degree in computer science from the university of cairo,  and i have been working in the field of artificial intelligence for over 5 years.",
    "i am a software engineer with a passion for natural language processing.  i have a degree  in computer science from the university of cairo,  and i have been working in the field of artificial  intelligence for over 5 years.",
    "my age is 15  years old,  and i am a student at the university of cairo,   studying computer science,   and i am interested in artificial intelligence.",
    "my faculty  is the faculty of computer science and engineering,  and my department is the department of artificial intelligence.",
    "my name is ali hassan , and i love  playing football, and i like   listening to music,  and i am a student at the university of cairo.",]


# Perform analyses
for text in texts:
  bigram_matrix, bigram_features = bigram_analysis(text)
  trigram_matrix, trigram_features = trigram_analysis(text)
  entities = chick_entities(text)

  print("Bigram Matrix:")
  print(bigram_matrix)
  print("Feature Names (Bigrams):")
  print(bigram_features)

  print("\nTrigram Matrix:")
  print(trigram_matrix)
  print("Feature Names (Trigrams):")
  print(trigram_features)

  print("\nNamed Entities:")
  print(entities)