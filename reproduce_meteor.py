import nltk
from nltk.translate.meteor_score import meteor_score

# Download wordnet if not present (usually handled by NLTK but good to be safe)
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
candidate = "Curriculum is stale."
reference = "History lessons can be used as state-sponsored propaganda, distorting the events of the past"

# Tokenization as per MeteorRuleEvaluator
candidate_tokens = candidate.split()
reference_tokens = reference.split()

print(f"Candidate tokens: {candidate_tokens}")
print(f"Reference tokens: {reference_tokens}")

score = meteor_score([reference_tokens], candidate_tokens)
print(f"METEOR Score: {score}")

# Try without punctuation to see if it matters
import re
def clean(text):
    return re.sub(r'[^\w\s]', '', text).split()

c_clean = clean(candidate)
r_clean = clean(reference)
print(f"Cleaned Candidate: {c_clean}")
print(f"Cleaned Reference: {r_clean}")
score_clean = meteor_score([r_clean], c_clean)
print(f"Cleaned METEOR Score: {score_clean}")
