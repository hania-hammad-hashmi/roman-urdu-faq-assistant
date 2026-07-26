# Roman Urdu/Saraiki FAQ Assistant

## Problem
Pakistan still has millions of children out of school, and even among those
enrolled, many families can't afford after-school tutoring to fill the gaps. Over
four years volunteering in community education, I kept seeing the same pattern:
students with basic, repeated questions and no one available to answer them
outside class hours. During my internship at AI Uplift, I worked on the data
side of a chatbot built for underprivileged communities, and it left me with a
question I wanted to answer myself, outside of work: could a much smaller
version of that idea, built by me alone, actually hold up? I built this project
afterward, independently, to find out.

## Dataset
This dataset (120 rows, 10 categories, 12 rows each) was written by me for
demonstration purposes. It is NOT real data collected from students — it was
self-authored to protect the privacy of the community I work with. It represents
the kinds of questions students commonly ask.

## How to Run
Requirements: Python 3.12, pandas, scikit-learn (all pre-installed in Google
Colab by default — no separate requirements.txt needed for a project this size).

1. Open a new Google Colab notebook.
2. Upload `sample_data.csv` to a `data/` folder in your Colab session.
3. Upload all `.py` files from `src/` to your Colab session (same root level
   as `data/`).
4. Run: `!python src/load_data.py` (expected output: confirms 120 rows loaded,
   takes under 1 second).
5. Run: `!python src/demo.py` for an interactive session, or any of the other
   `src/` scripts individually to see specific test results.

## v1: Rule-Based Matching (NOT machine learning)
`src/matcher_v1.py` uses Python's built-in `difflib` library to compare a user's
question against every question in the dataset and return the answer for the
closest text match. This is fuzzy string matching, not AI/ML — it has no
learning component and doesn't generalize beyond comparing text similarity.

It also includes a minimum confidence threshold (0.5) — if the best match's
similarity score falls below this, the tool says it doesn't know rather than
guessing. This isn't perfect: testing showed a borderline question can still
score just above the threshold and return a topically related but incorrect
answer.

## v2: Machine Learning Classifier
`src/model_v2.py` uses TF-IDF to convert questions into numeric vectors, then
trains a Logistic Regression classifier to predict the category of a new
question. This is genuine, if small-scale, machine learning.

## Known Limitations
- **v1 (fuzzy matching)** uses a min_confidence threshold (0.5) to avoid
  guessing wrong. Testing across multiple rounds found this only partially
  works: on Day 6, 2 of 3 unrelated test questions still passed the threshold
  with wrong answers. On Day 10, a water-related question was incorrectly
  matched to an unrelated "old books" question at a 0.57 score — still above
  threshold. The confidence score does not reliably reflect topical
  correctness.

- **v2 (ML classifier)** has no confidence threshold at all — it always
  predicts the closest known category. A direct comparison (Day 8) showed v1
  correctly refusing 3 unrelated questions (scores 0.48/0.37/0.45) while v2
  confidently answered all 3 wrong. However, v2 is not uniformly worse: on Day
  10, v2 correctly identified the "water" category for a question v1 had
  matched incorrectly, likely because it learned word patterns during training
  rather than relying on a single closest-question comparison.

- **Investigated the "transport" prediction issue (see
  `src/debug_transport_bias.py`).** Initially this looked like an unexplained
  bias. Investigation ruled out two hypotheses: it's not caused by class
  imbalance (all 10 categories have exactly 12 rows each) and not caused by
  generic overlapping words (transport's top learned words — "van," "parking,"
  "area" — are specific and reasonable). The actual cause: the model reaches
  99.2% accuracy on its own training data (119/120 — measured on training data
  itself, not a held-out test set, since the goal here was diagnosing
  overfitting rather than claiming generalization) but produces very
  low-confidence, near-random predictions on unfamiliar phrasing — all
  candidate categories cluster around 11-13% probability for the failing
  questions, essentially a coin-flip among close options. This is a sign of
  overfitting: with only 120 examples, the model appears to memorize specific
  training questions rather than learn patterns that generalize to new
  phrasing.

- **v2 has no handling for empty input** and was not tested against it.

- **Overall:** neither version reliably distinguishes "confident and correct"
  from "confident and wrong." v1 fails by passing a bad match through its
  threshold; v2 fails by having no threshold at all and, as confirmed above,
  by overfitting to a small training set. Fixing this properly would require a
  larger dataset, a confidence mechanism for v2, and a held-out validation set
  to actually measure generalization — see "What a v3 Would Need" below.

## What a v3 Would Need
A meaningful next version would require: (1) a larger dataset — likely 500+
rows to reduce the overfitting shown above, (2) a proper train/test split to
measure real generalization instead of training accuracy, and (3) a confidence
threshold for v2, similar to v1's, so it can say "I don't know" instead of
always guessing.

## Example Input/Output

**Example 1 — v1 correctly finds a fuzzy match:**
Input: "mein apna unifrom kahaan sey ley sakta hoon"
v1 match: "winter uniform kab se lagta hai" (confidence: 0.59)
Answer: "November se winter uniform shuru hota hai"

**Example 2 — v1 correctly refuses; v2 guesses wrong:**
Input: "ya larkon ki fees aur akrkion ki fees same hai"
v1: "Sorry, I'm not confident I understood that question." (confidence: 0.48)
v2: predicted category "fees" -> "Har month ki das tareekh tak fees jama karni hai"
(Note: this question is not actually about fees — it asks whether boys' and
girls' fees are the same, a comparison question v2 mishandled.)

**Example 3 — v1 fails, v2 succeeds (the reversal found on Day 10):**
Input: "agar gatar ka pani bhar jaye toh kya karna chahiay"
v1 match: "agar paani khatam ho jaye to kya karen" (confidence: 0.66)
-> "Office se refill karwa sakte ho" (wrong — this is about water running OUT,
not flooding drains)
v2: predicted category "water" -> "School ke pass water filter plant hai"
(v2's answer is also imperfect, but the category prediction was more relevant
than v1's specific match)

**Example 4 — the transport prediction issue (see Known Limitations for the
confirmed cause: overfitting, not bias):**
Input: "test kitnay marka ka hai" (how many marks is the test out of)
v2: predicted category "transport" -> "School office se van book karwa sakte ho"
Input: "syllabus mein kya kya aa raha hai" (what's covered in the syllabus)
v2: predicted category "transport" -> "School office se van book karwa sakte ho"
(Neither question relates to transport. Probability breakdown showed both
predictions were low-confidence near-ties, not strong wrong guesses.)

## What I Learned
Most of what I actually learned building this had nothing to do with the code
itself. I broke my GitHub setup more times than I could count - clone errors, a
merge conflict I had to resolve by hand, a security token I accidentally
exposed and had to revoke within minutes of realizing it. After that, I stopped
ever typing a token directly into a code cell — I only ever entered it through
getpass's hidden input field from that point on. None of it was elegant. But
debugging git turned out to teach me the same lesson as debugging the model:
check what actually happened, not what you assumed happened. I made that
mistake with my own code too - after finding that my rule-based version
correctly refused questions it didn't recognize while my trained model guessed
wrong every time, I started assuming the simple version was just better. Then I
tested a real question about running water, and the simple version failed it,
matching it to an unrelated question about old textbooks - while the trained
model got it right. Neither version was consistently trustworthy. I'd built two
different, specific ways of being wrong, not one good tool and one bad one.
That contradicted the clean story I'd already started telling myself about the
project, and I had to sit with that instead of editing it away.
