import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import json

class SimpleChatBot:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.data = []
        self.responses = []
        self.load_data()

    def load_data(self):
        try:
            with open('chat_data.json', 'r') as file:
                data = json.load(file)
                self.data = data['questions']
                self.responses = data['responses']
                self.train()
        except FileNotFoundError:
            print("No training data found. Starting fresh.")

    def save_data(self):
        data = {
            'questions': self.data,
            'responses': self.responses
        }
        with open('chat_data.json', 'w') as file:
            json.dump(data, file)

    def train(self):
        X = self.vectorizer.fit_transform(self.data)
        y = np.array(self.responses)
        self.classifier.fit(X, y)

    def respond(self, user_input):
        user_input_transformed = self.vectorizer.transform([user_input])
        prediction = self.classifier.predict(user_input_transformed)
        return prediction[0]

    def learn(self, user_input, response):
        self.data.append(user_input)
        self.responses.append(response)
        self.train()
        self.save_data()

# Create a chatbot instance
chatbot = SimpleChatBot()

# Chatbot interaction loop
print("Chatbot is ready! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = chatbot.respond(user_input)
    print(f"Chatbot: {response}")

    # Optionally, ask the user for feedback to learn new responses
    feedback = input("Is this response correct? (yes/no) ")
    if feedback.lower() == 'no':
        correct_response = input("Please provide the correct response: ")
        chatbot.learn(user_input, correct_response)
        print("Thanks! I have learned the new response.")
