"""
matcher_v1.py
v1 of the FAQ assistant: rule-based fuzzy string matching.
This does NOT use machine learning. It compares the incoming question
against every question in the dataset and returns the answer for the
closest text match, using Python's built-in difflib library.
"""

import difflib
from load_data import load_faq_data


def find_best_match(user_question, data, min_confidence=0.5):
    """
    It compares the queation against the acctual data set, then returns the answer 
    that top at the similarity scores. If the match is below the min confidence , then it returns 
    no match, instead of gussing wrong "
    """
    if user_question.strip() == "":
        return None, "Please type a question.", 0.0

    questions = data["question"].tolist()
    matches = difflib.get_close_matches(user_question, questions, n=1, cutoff=0.0)

    if not matches:
        return None, "Sorry, no matching question found.", 0.0

    best_question = matches[0]
    score = round(difflib.SequenceMatcher(None, user_question, best_question).ratio(), 2)

    if score < min_confidence:
        return None, "Sorry, I'm not confident I understood that question.", score

    answer_row = data[data["question"] == best_question].iloc[0]
    return best_question, answer_row["answer"], score
