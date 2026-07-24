"""
model_v2.py
v2 of the FAQ assistand is machine learing . It uses TF-IDF to convert questions into numbers,
then it trains the Logistic Regression classifier to predict the catagoryof a question. from the given dats set .
Honest limitation: with only 10 to 60 rows, this is learning from a
very small dataset hence,not comparable to production ML.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from load_data import load_faq_data


def train_classifier(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data["question"])
    y = data["category"]
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model


def predict_category(user_question, vectorizer, model):
    X_new = vectorizer.transform([user_question])
    return model.predict(X_new)[0]


def get_answer_by_category(category, data):
    matches = data[data["category"] == category]
    if len(matches) == 0:
        return "Sorry, no answer found for this category."
    return matches.iloc[0]["answer"]


if __name__ == "__main__":
    data = load_faq_data("data/sample_data.csv")
    vectorizer, model = train_classifier(data)
    test_questions = ["class room ki seating kya hogi", "agar mein uniform mein na aaon toh kya hoga","meray baba mujhay maarat hein subha","mujhay class mein neend aati hai"]
    for q in test_questions:
        category = predict_category(q, vectorizer, model)
        answer = get_answer_by_category(category, data)
        print(f"Input: {q}\nPredicted category: {category}\nAnswer: {answer}\n---")
