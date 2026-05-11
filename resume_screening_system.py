import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/resume.csv")

# Job description
job_description = "Python Machine Learning Data Science"

# Combine job description with resume skills
documents = [job_description] + df["Skills"].tolist()

# Convert text into vectors
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(documents)

# Calculate similarity
similarity = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

# Add scores
df["Score"] = similarity

# Rank candidates
ranked_df = df.sort_values(by="Score", ascending=False)

# Print results
print("\nCandidate Ranking:\n")
print(ranked_df)

# Plot graph
plt.bar(ranked_df["Name"], ranked_df["Score"])
plt.title("Resume Screening Scores")
plt.xlabel("Candidates")
plt.ylabel("Matching Score")
plt.xticks(rotation=45)

# Save graph
plt.savefig("images/sample_output.png")

plt.show()