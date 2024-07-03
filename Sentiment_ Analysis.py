import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset for sentiment analysis
df_sentiment = pd.read_csv('classified_reviews_with_sentiment.csv')

# Preprocess: Remove rows where either 'Translated_Review' or 'Sentiment' is NaN
df_sentiment.dropna(subset=['Translated_Review', 'sentiment'], inplace=True)

# Split the dataset into training and testing sets for sentiment analysis
X_train, X_test, y_train, y_test = train_test_split(df_sentiment['Translated_Review'], df_sentiment['sentiment'], test_size=0.2, random_state=42)

# Create a text processing and classification pipeline
pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter to handle convergence
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print(f'Sentiment Analysis Accuracy: {accuracy_score(y_test, y_pred)}')

# Load the dataset for AI-generated text classification
df_classification = pd.read_csv('classified_reviews_with_sentiment.csv')

# Preprocess: Remove rows where 'Translated_Review' is NaN
df_classification.dropna(subset=['Translated_Review', 'sentiment'], inplace=True)

# Assuming there's a column 'Is_AI_Generated' that indicates if the text is AI-generated or not
# For now, let's create a dummy 'Is_AI_Generated' column for demonstration
# You should replace this with your actual column
df_classification['Is_AI_Generated'] = df_classification['sentiment'] > 0.5  # Example condition, adjust as needed

# Split the dataset into training and testing sets for AI-generated text classification
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(df_classification['Translated_Review'], df_classification['Is_AI_Generated'], test_size=0.2, random_state=42)

# Train the model for AI-generated text classification
pipeline_cls = make_pipeline(
    TfidfVectorizer(stop_words='english'),
    LogisticRegression(random_state=42, max_iter=1000)
)

# Train the model
pipeline_cls.fit(X_train_cls, y_train_cls)

# Evaluate the model
y_pred_cls = pipeline_cls.predict(X_test_cls)
print(f'AI-Generated Text Classification Accuracy: {accuracy_score(y_test_cls, y_pred_cls)}')

# Load the dataset for clustering
# Assuming the same dataset is used, we can continue from the preprocessed df_sentiment for a consistent approach
df_clustering = df_sentiment.copy()

# Ensure no NaN values in 'Translated_Review' for clustering
df_clustering['Translated_Review'].fillna('', inplace=True)  # Filling NaN with empty strings

# Convert texts to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df_clustering['Translated_Review'])

# Preliminary analysis to choose an optimal number of clusters (optional)
silhouette_scores = []
for n_clusters in range(2, 7):  # Example: testing between 2 to 6 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    silhouette_scores.append(score)
    print(f"Silhouette Score for {n_clusters} clusters: {score}")

# Plot silhouette scores (optional)
plt.plot(range(2, 7), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')
plt.show()

# Assuming you decide on an optimal number of clusters based on the analysis
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Automatically selecting the best number of clusters

# Clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(X)
df_clustering['cluster'] = clusters

# Inspect texts in each cluster to infer AI-generated vs. human-generated
for i in range(optimal_clusters):
    print(f"\nCluster {i} samples:")
    print(df_clustering[df_clustering['cluster'] == i]['Translated_Review'].head(10).to_string(index=False), '\n')





