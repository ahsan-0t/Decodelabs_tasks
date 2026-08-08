# AI Recommendation Logic — Tech Stack Recommender

**Project 3 – DecodeLabs AI/ML Industrial Training Kit (Batch 2026)**
**Author:** Ahsan Tahir

## 📌 Overview
A content-based recommendation engine that maps a user's raw skills and
interests to the most relevant tech career paths (e.g., Data Scientist,
DevOps Engineer, Frontend Developer). This is the same core logic behind
real-world recommenders like Netflix or Amazon, applied here to a
`job_role → skills` dataset instead of movies or products.

## 🎯 Goal
Build a simple recommendation system that:
- Takes user input (skills/interests)
- Matches preferences to items using similarity logic (not random guessing)
- Displays the Top-N most relevant recommendations

## ⚙️ How It Works (Input → Process → Output)
1. **Input** — 16 job roles are loaded from `raw_skills.csv`, each tagged with its required skills. The user provides 3+ of their own skills.
2. **Process (Content-Based Filtering)**
   - **Vector Mapping:** Both job-role skill lists and the user's skills are converted into numerical vectors using **TF-IDF** (Term Frequency–Inverse Document Frequency), so specific, descriptive skills (e.g., "Kubernetes") count for more than generic ones, and everything maps into the same shared vocabulary.
   - **Scoring:** **Cosine Similarity** is calculated between the user's vector and every job role's vector — this measures the *angle* between them rather than raw overlap count, so it isn't biased by how many skills a role happens to list.
   - **Sorting & Filtering:** Roles are sorted by score (highest first), zero-score (irrelevant) roles are dropped, and the list is truncated to the Top 3 — preventing "choice overload."
3. **Output** — The Top 3 matching career paths are displayed with a match percentage and the role's key skills.

## 🧊 Handling the "Cold Start" Problem
If a user's skills don't overlap with anything in the dataset's vocabulary
(e.g., "Cooking, Photography"), all similarity scores are 0. Rather than
showing irrelevant 0% matches, the program detects this and shows a
friendly message suggesting more common tech skills instead.

## ▶️ How to Run
```bash
pip install pandas scikit-learn
python3 recommender.py
```
Includes 3 built-in demo runs. To try your own skills, edit the calls at
the bottom of `recommender.py`, e.g.:
```python
recommend(["Python", "Machine Learning", "SQL"])
```

## 📂 Files
- `recommender.py` — full recommendation engine (TF-IDF + Cosine Similarity)
- `raw_skills.csv` — dataset of 16 job roles and their associated skills
- `README.md` — this file

## 🚀 Key Skills Demonstrated
Content-based filtering, TF-IDF vectorization, cosine similarity, logic
building, pattern matching, and handling real-world edge cases (cold
start / no-match scenarios).
