"""
debug_transport_bias.py
Investigates why v2 (the ML classifier) sometimes predicts "transport"
for unrelated questions. Runs three separate checks.
"""

import numpy as np
from load_data import load_faq_data
from model_v2 import train_classifier, predict_category


def check_1_top_words_per_category(vectorizer, model):
    """
    Shows which words the model learned are most strongly associated
    with EACH category, not just transport. If "transport" has unusually
    generic top words (like "hai" or "kya" instead of specific words like
    "van" or "bus"), that's a real clue.
    """
    print("=== CHECK 1: Top words per category ===\n")
    feature_names = vectorizer.get_feature_names_out()
    for category in model.classes_:
        idx = list(model.classes_).index(category)
        coefficients = model.coef_[idx]
        top_indices = np.argsort(coefficients)[-5:]
        top_words = [feature_names[i] for i in reversed(top_indices)]
        print(f"{category}: {top_words}")
    print()


def check_2_specific_failing_inputs(data, vectorizer, model):
    """
    Takes the exact questions that were misclassified as "transport"
    and shows the FULL probability breakdown across all categories,
    not just the top prediction. This shows whether "transport" barely
    won, or won by a large margin.
    """
    print("=== CHECK 2: Probability breakdown for known failing inputs ===\n")
    failing_questions = [
        "test kitnay marka ka hai",
        "syllabus mein kya kya aa raha hai",
    ]
    for q in failing_questions:
        X_new = vectorizer.transform([q])
        probabilities = model.predict_proba(X_new)[0]
        print(f"Input: {q}")
        for category, prob in sorted(zip(model.classes_, probabilities), key=lambda x: -x[1])[:3]:
            print(f"  {category}: {prob:.3f}")
        print()


def check_3_train_accuracy(data, vectorizer, model):
    """
    Checks accuracy on the SAME data the model was trained on.
    Note: this is not a true test of generalization (that would require
    a train/test split), but it tells us whether the model is even
    correctly fitting the data it has seen.
    """
    print("=== CHECK 3: Accuracy on training data itself ===\n")
    X = vectorizer.transform(data["question"])
    predictions = model.predict(X)
    correct = sum(predictions == data["category"])
    total = len(data)
    print(f"Correct: {correct}/{total} ({100*correct/total:.1f}%)")
    print("(Note: high accuracy here does NOT prove the model generalizes")
    print("well to new phrasing - it only shows it fits the training data.)\n")


if __name__ == "__main__":
    data = load_faq_data("data/sample_data.csv")
    vectorizer, model = train_classifier(data)

    check_1_top_words_per_category(vectorizer, model)
    check_2_specific_failing_inputs(data, vectorizer, model)
    check_3_train_accuracy(data, vectorizer, model)
