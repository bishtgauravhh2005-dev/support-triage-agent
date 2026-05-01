import pandas as pd
import re

# ===============================
# 1️⃣ Load Dataset
# ===============================
df = pd.read_csv("support_tickets.csv")
df.columns = df.columns.str.strip().str.lower()

print("Columns:", df.columns)

# Safe column detection
def get_issue(row):
    for col in ["issue", "ticket_text", "text"]:
        if col in row:
            return str(row.get(col, ""))
    return ""

# ===============================
# 2️⃣ Support Corpus
# ===============================
documents = [
    "Coding test not loading fix by refreshing browser or checking internet",
    "Assessment issues due to browser compatibility or network problems",
    "Submission errors happen due to timeouts or incorrect execution",
    "Claude API usage and billing details available in dashboard",
    "API not working check API key and quota",
    "Billing issues depend on subscription plan",
    "Unauthorized transactions must be reported to bank immediately",
    "Payment failures due to insufficient balance or restrictions",
    "Refunds should be requested via merchant or bank",
    "Reset password using account settings",
    "Login issues solved using forgot password option"
]

# ===============================
# 3️⃣ Retrieval (Embedding + Fallback)
# ===============================
use_embeddings = True

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(documents)

    def retrieve(query):
        query_embedding = model.encode([query])
        scores = cosine_similarity(query_embedding, doc_embeddings)
        idx = scores.argmax()
        return documents[idx], float(scores[0][idx])

except Exception as e:
    print("⚠️ Embedding model failed, using TF-IDF fallback:", e)
    use_embeddings = False

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)

    def retrieve(query):
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, doc_vectors)
        idx = scores.argmax()
        return documents[idx], float(scores[0][idx])

# ===============================
# 4️⃣ Helper Functions
# ===============================
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", "", text.lower())

def classify_request(issue):
    issue = issue.lower()

    if any(x in issue for x in ["error", "failed", "not working", "problem", "issue"]):
        return "bug"
    
    if any(x in issue for x in ["feature", "add", "improve", "request"]):
        return "feature_request"
    
    return "product_issue"

def detect_product_area(issue):
    text = issue.lower()

    if any(x in text for x in ["payment", "card", "refund", "charged"]):
        return "payments"

    if any(x in text for x in ["fraud", "unauthorized", "suspicious"]):
        return "fraud"

    if any(x in text for x in ["login", "password", "account"]):
        return "account_access"

    if any(x in text for x in ["test", "assessment", "submission"]):
        return "assessments"

    if "api" in text:
        return "api_usage"

    if any(x in text for x in ["billing", "subscription"]):
        return "billing"

    return "general"

def detect_risk(issue):
    issue = issue.lower()

    if any(x in issue for x in ["fraud", "unauthorized", "charged", "hacked"]):
        return "high"

    if "urgent" in issue:
        return "medium"

    return "low"

def split_issue(issue):
    return re.split(r"\.|\band\b|\?|,", issue)

# ===============================
# 5️⃣ MAIN PIPELINE
# ===============================
results = []

for _, row in df.iterrows():
    issue = get_issue(row)

    if not issue.strip():
        results.append({
            "status": "escalated",
            "product_area": "general",
            "response": "The request is unclear and has been escalated.",
            "justification": "Empty input",
            "request_type": "invalid"
        })
        continue

    parts = split_issue(issue)

    best_score = 0
    best_context = ""
    final_type = "product_issue"
    final_area = "general"
    max_risk = "low"

    for part in parts:
        part = clean_text(part)
        if not part:
            continue

        req_type = classify_request(part)
        area = detect_product_area(part)
        risk = detect_risk(part)
        context, score = retrieve(part)

        if score > best_score:
            best_score = score
            best_context = context
            final_type = req_type
            final_area = area

        if risk == "high":
            max_risk = "high"

    # ===============================
    # DECISION LOGIC
    # ===============================
    if max_risk == "high":
        status = "escalated"
    elif best_score < 0.28:
        status = "escalated"
    else:
        status = "replied"

    # ===============================
    # RESPONSE
    # ===============================
    if status == "replied":
        response = (
            f"According to support documentation: {best_context}. "
            "If the issue persists, please contact support."
        )
    else:
        response = (
            "This issue may involve sensitive or unclear information and has been escalated to human support."
        )

    # ===============================
    # JUSTIFICATION
    # ===============================
    justification = (
        f"type={final_type}, area={final_area}, risk={max_risk}, confidence={round(best_score,2)}"
    )

    results.append({
        "status": status,
        "product_area": final_area,
        "response": response,
        "justification": justification,
        "request_type": final_type
    })

# ===============================
# 6️⃣ SAVE OUTPUT
# ===============================
pd.DataFrame(results).to_csv("output.csv", index=False)

print("OUTPUT GENERATED!")