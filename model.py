import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
from textblob import TextBlob
import matplotlib.pyplot as plt

# PARAMETERS
dataset_path = 'DatingAppReviewsDataset.csv'  # Path to your dataset
text_column_name = 'Review'  # Column with text data
label_column_name = 'Rating'  # Column with labels for classification

# Load the dataset
df = pd.read_csv(dataset_path)

# Preprocess: Remove NaNs
df.dropna(subset=[text_column_name, label_column_name], inplace=True)

# Convert text column to string
df[text_column_name] = df[text_column_name].astype(str)

# Sentiment Analysis with TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['sentiment'] = df[text_column_name].apply(get_sentiment)
# You can further categorize sentiment values into positive, neutral, or negative based on your criteria

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[text_column_name], df[label_column_name], test_size=0.2, random_state=42)

# Text Processing and Classification Pipeline
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=1000),
    LogisticRegression(random_state=42, max_iter=1000)
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(f'Classification Accuracy: {accuracy_score(y_test, y_pred)}')

# Note: For AI-generated text classification, you'd typically need a labeled dataset and a model trained to distinguish AI from human text

# Clustering (ensure no NaN values in text column)
df[text_column_name].fillna('', inplace=True)

# Convert texts to TF-IDF features for clustering
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df[text_column_name])

# Silhouette analysis for optimal cluster number
silhouette_scores = []
for n_clusters in range(2, 7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    silhouette_scores.append(score)
    print(f"Silhouette Score for {n_clusters} clusters: {score}")

# Plot silhouette scores
plt.plot(range(2, 7), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')
plt.show()

# Choose optimal number of clusters based on silhouette scores
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2

# Perform clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters

# Inspect texts in each cluster
for i in range(optimal_clusters):
    print(f"\nCluster {i} samples:")
    print(df[df['cluster'] == i][text_column_name].head(10).to_string(index=False), '\n')

# Speculative step: Assigning labels based on cluster analysis
df['predicted_label'] = df['cluster'].apply(lambda x: 'AI Generated' if x == 0 else 'Human Generated')

# Save the updated dataset with clusters and speculative labels
df.to_csv('classified_reviews_with_sentiment.csv', index=False)
