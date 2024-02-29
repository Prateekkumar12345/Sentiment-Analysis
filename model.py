# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.pipeline import Pipeline

# # Load the labeled dataset
# df_labeled = pd.read_csv('app_details.csv')
# texts = df_labeled['summary']
# labels = df_labeled['reviews']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Create a text classification pipeline
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words='english')),
#     ('classifier', LogisticRegression(random_state=42)),
# ])

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# predictions = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy: {accuracy}')
# print(confusion_matrix(y_test, predictions))

# # Load the unlabeled dataset
# df_unlabeled = pd.read_csv('app_details.csv')
# unlabeled_texts = df_unlabeled['summary']  # Assuming the column with texts is named 'text'

# # Predict the labels for the unlabeled dataset
# predicted_labels = pipeline.predict(unlabeled_texts)

# # Add the predicted labels to the DataFrame
# df_unlabeled['predicted_label'] = predicted_labels
# df_unlabeled['predicted_label'] = df_unlabeled['predicted_label'].apply(lambda x: 'Human-Generated' if x == 1 else 'AI-Generated')

# # Save or use the classified dataset
# df_unlabeled.to_csv('app_details.csv', index=False)


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

# # Load your labeled dataset
# df = pd.read_csv('googleplaystore_user_reviews.csv')  # Update the file name to your labeled dataset file

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['Translated_Review'], df['Sentiment'], test_size=0.2, random_state=42)

# # Create a text processing and classification pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english'),
#     LogisticRegression(random_state=42)
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# # Assuming you have an unlabeled dataset that you want to classify
# unlabeled_df = pd.read_csv('googleplaystore_user_reviews.csv')  # Update the file name to your unlabeled dataset file
# unlabeled_texts = unlabeled_df['Translated_Review']  # Update 'text' to the name of the column containing text in your dataset

# # Predict whether the text is AI-generated (0) or human-generated (1)
# predicted_labels = pipeline.predict(unlabeled_texts)

# # Add the predictions to your unlabeled dataset
# unlabeled_df['predicted_label'] = predicted_labels
# unlabeled_df['predicted_label'] = unlabeled_df['predicted_label'].apply(lambda x: 'AI-Generated' if x == 0 else 'Human-Generated')

# # Save or further process your now labeled dataset
# unlabeled_df.to_csv('classified_dataset.csv', index=False)




# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv('Reviews.csv')  # Update the file path
# texts = df['Review Text']  # Update the column name to your text column

# # Convert texts to TF-IDF features
# vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# X = vectorizer.fit_transform(texts)

# # Use KMeans to cluster the data
# # Choosing 2 clusters for AI vs. human might be oversimplifying, adjust as necessary
# kmeans = KMeans(n_clusters=2, random_state=42)
# clusters = kmeans.fit_predict(X)

# # Add cluster information to the dataframe
# df['cluster'] = clusters

# # Optionally, visualize the clusters (if reduced to 2D using PCA, for example)
# # This step is skipped here as it's more complex and not always informative for high-dimensional text data

# # Explore the clusters to try to identify characteristics of AI-generated vs. human-generated text
# # This is a manual step and requires inspection of the texts in each cluster
# print(df.groupby('cluster')['Review Text'].apply(lambda texts: '\n'.join(texts.iloc[:5])))

# # Based on your inspection, you might decide one cluster seems more like AI-generated text
# # This is a heuristic and subjective process without guarantees
# # For example, if you decide cluster 0 is AI-generated, you can label accordingly
# df['predicted_label'] = df['cluster'].apply(lambda x: 'AI-Generated' if x == 0 else 'Human-Generated')

# # Save or further process your dataset
# df.to_csv('clustered_dataset.csv', index=False)




# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv('Reviews.csv')  # Ensure the file path is correct
# texts = df['Review Text']  # Ensure the column name matches your dataset

# # Convert texts to TF-IDF features
# vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# X = vectorizer.fit_transform(texts)

# # Preliminary analysis to choose an optimal number of clusters (optional)
# # Note: This step can be computationally expensive for large datasets
# silhouette_scores = []
# for n_clusters in range(2, 7):  # Example: testing between 2 to 6 clusters
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(X)
#     score = silhouette_score(X, clusters)
#     silhouette_scores.append(score)
#     print(f"Silhouette Score for {n_clusters} clusters: {score}")

# # Plot silhouette scores (optional)
# plt.plot(range(2, 7), silhouette_scores, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score Analysis')
# plt.show()

# # Assuming you decide on an optimal number of clusters based on the analysis
# optimal_clusters = 2  # Update this based on your analysis

# # Clustering with the chosen number of clusters
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
# clusters = kmeans.fit_predict(X)
# df['cluster'] = clusters

# # Inspect texts in each cluster to infer AI-generated vs. human-generated
# # This requires manual inspection and interpretation
# for i in range(optimal_clusters):
#     print(f"\nCluster {i} samples:")
#     print(df[df['cluster'] == i]['Review Text'].head(100).to_string(index=False), '\n')

# # After inspection, if you identify which cluster might correspond to AI or human
# # Update the labeling accordingly - this is still speculative
# # Example decision: cluster 0 -> AI-Generated, cluster 1 -> Human-Generated
# df['predicted_label'] = df['cluster'].apply(lambda x: 'AI-Generated' if x == 0 else 'Human-Generated')

# # Save the updated dataset
# df.to_csv('classified_reviews.csv', index=False)







# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv('app_details.csv')  # Make sure your dataset has a properly labeled column
# Reviews = df['score']  # Text data
# Labels = df['ratings']  # Binary labels: 0 for AI-generated, 1 for Human-generated

# # Preprocess the text: TF-IDF Vectorization
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X = vectorizer.fit_transform(Reviews).toarray()

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, Labels, test_size=0.2, random_state=42)

# # Train a Logistic Regression model
# model = LogisticRegression(random_state=42, max_iter=1000)
# model.fit(X_train, y_train)

# # Predict on the test set
# predictions = model.predict(X_test)

# # Calculate the accuracy score
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy score: {accuracy}')

# # Confusion matrix
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title('Confusion Matrix')
# plt.show()


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Assuming 'app_details.csv' contains a column 'Reviews' with text to analyze
# df = pd.read_csv('app_details.csv')
# Reviews = df['reviews'].astype(str)  # Ensuring all data is treated as string

# # Preprocess the text: TF-IDF Vectorization
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X = vectorizer.fit_transform(Reviews)

# # Use PCA to reduce dimensions for visualization (optional, but helps in plotting)
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X.toarray())

# # Apply KMeans Clustering
# kmeans = KMeans(n_clusters=2, random_state=42)
# clusters = kmeans.fit_predict(X_reduced)

# # Plot the clustered data
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
# plt.title('Clusters of Reviews')
# plt.xlabel('PCA Feature 1')
# plt.ylabel('PCA Feature 2')
# plt.colorbar(label='Cluster')
# plt.show()

# # Adding cluster information to the DataFrame
# df['Cluster'] = clusters

# # You now need to manually inspect texts in each cluster to speculate about their nature
# print(df.groupby('Cluster')['reviews'].apply(lambda x: '\n'.join(x.iloc[:5])))

# # Save the DataFrame with cluster assignments if needed
# df.to_csv('clustered_app_details.csv', index=False)




# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Load your dataset
# df = pd.read_csv('app_details.csv')
# Reviews = df['reviews'].astype(str)  # Ensuring all data is treated as string

# # Convert texts to TF-IDF features
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X = vectorizer.fit_transform(Reviews)

# # Use PCA to reduce dimensions for visualization (Optional)
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X.toarray())

# # Apply KMeans Clustering
# kmeans = KMeans(n_clusters=2, random_state=42)
# clusters = kmeans.fit_predict(X_reduced)

# # Add the cluster info to the DataFrame
# df['cluster'] = clusters

# # Optionally, plot the clustered data for a visual overview
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
# plt.title('Clusters of Reviews')
# plt.xlabel('PCA Feature 1')
# plt.ylabel('PCA Feature 2')
# plt.colorbar(label='Cluster')
# plt.show()

# # Manually inspect clusters to infer AI-generated vs. human-generated
# for cluster_id in [0, 1]:
#     print(f"\nCluster {cluster_id} samples:")
#     print(df[df['cluster'] == cluster_id]['reviews'].head(10).values, '\n')




# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv('app_details.csv')
# reviews = df['reviews'].astype(str)  # Assuming 'reviews' column exists

# # Convert texts to TF-IDF features
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X = vectorizer.fit_transform(reviews)

# # Use PCA to reduce dimensions for visualization (optional)
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X.toarray())

# # Apply KMeans Clustering
# kmeans = KMeans(n_clusters=2, random_state=42)
# clusters = kmeans.fit_predict(X_reduced)

# # Plot the clustered data for a visual overview (optional)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
# plt.title('Clusters of Reviews')
# plt.xlabel('PCA Feature 1')
# plt.ylabel('PCA Feature 2')
# plt.colorbar(label='Cluster')
# plt.show()

# # Add the cluster info to the DataFrame
# df['cluster'] = clusters

# # Manually inspect clusters to infer AI-generated vs. human-generated
# # This step is subjective and requires manual work
# print("\nCluster 0 samples:")
# print(df[df['cluster'] == 0]['reviews'].head(10).values)
# print("\nCluster 1 samples:")
# print(df[df['cluster'] == 1]['reviews'].head(10).values)




# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv('app_details.csv')  # Adjust the filename and column names as needed
# Reviews = df['summary']  # Assuming 'Review Text' is the column with text data

# # Preprocess the text: TF-IDF Vectorization
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X = vectorizer.fit_transform(Reviews).toarray()

# # Dimensionality Reduction (Optional)
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X)

# # Clustering with KMeans
# kmeans = KMeans(n_clusters=2, random_state=42)
# clusters = kmeans.fit_predict(X_reduced)

# # Add cluster info to your DataFrame
# df['cluster'] = clusters

# # Plot the clustered data for a visual overview
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6)
# plt.title('Clusters of Reviews')
# plt.xlabel('PCA Feature 1')
# plt.ylabel('PCA Feature 2')
# plt.colorbar(label='Cluster')
# plt.show()

# # Assuming after manual inspection of clusters:
# # Let's speculate: cluster 0 -> AI-Generated, cluster 1 -> Human-Generated
# # Note: The actual cluster numbers for AI/Human might differ based on your inspection
# df['Speculative Classification'] = df['cluster'].apply(lambda x: 'AI-Generated' if x == 0 else 'Human-Generated')

# # Print a summary and samples for manual verification
# print(df.groupby('Speculative Classification')['reviews'].count())

# print("\nSample AI-Generated Reviews:")
# print(df[df['Speculative Classification'] == 'AI-Generated']['reviews'].head(100), '\n')

# print("Sample Human-Generated Reviews:")
# print(df[df['Speculative Classification'] == 'Human-Generated']['reviews'].head(100), '\n')

# # Optionally, if you have labeled data for training and testing:
# # Assuming 'Rating' is a binary label indicating AI (0) or Human (1) - adjust as needed
# if 'Rating' in df.columns:
#     # Split the dataset into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_reduced, df['summary'], test_size=0.2, random_state=42)

#     # Train a Logistic Regression model
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)

#     # Predict on the test set
#     predictions = model.predict(X_test)

#     # Calculate the accuracy score
#     accuracy = accuracy_score(y_test, predictions)
#     print(f'Accuracy score: {accuracy}')
# else:
#     print("none")


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv('Reviews.csv')
# Reviews = df['Review Text']  # Adjust column name as needed
# Labels = df['Rating']  # Adjust column name as needed for labels

# # Enhanced Preprocessing: TF-IDF Vectorization without immediate PCA reduction
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(Reviews).toarray()

# # Optionally, keep PCA to reduce dimensionality, but consider evaluating its impact
# # pca = PCA(n_components=2)
# # X_reduced = pca.fit_transform(X)

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, Labels, test_size=0.2, random_state=42)

# # Model Tuning: Experiment with different values of C to find optimal setting
# best_accuracy = 0
# best_c = 1
# for c in [0.01, 0.1, 1, 10, 100]:
#     model = LogisticRegression(C=c, random_state=42)
#     model.fit(X_train, y_train)
#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_c = c

# # Train the model with the best C value found
# model = LogisticRegression(C=best_c, random_state=42)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)

# # Confusion Matrix Visualization
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.title(f'Confusion Matrix (Accuracy: {accuracy:.4f}, C: {best_c})')
# plt.show()

# print(f'Best C value: {best_c}')
# print(f'Accuracy score: {accuracy}')



# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV

# # Load your dataset
# df = pd.read_csv('Reviews.csv')
# Reviews = df['Review Text']
# Labels = df['Rating']  # Ensure this column exists and is correctly populated

# # Preprocess the text: TF-IDF Vectorization with bi-grams
# vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
# X = vectorizer.fit_transform(Reviews).toarray()

# # Optimize PCA Components based on explained variance
# pca = PCA(n_components=0.95)  # Adjust based on variance explanation
# X_reduced = pca.fit_transform(X)

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_reduced, Labels, test_size=0.2, random_state=42)

# # Hyperparameter tuning for Logistic Regression
# param_grid = {
#     'C': [0.01, 0.1, 1, 10],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear', 'saga']
# }

# model = LogisticRegression(random_state=42, max_iter=10000)
# clf = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
# clf.fit(X_train, y_train)

# # Best model after grid search
# best_model = clf.best_estimator_

# # Predict on the test set with the best model
# predictions = best_model.predict(X_test)

# # Calculate the accuracy score
# accuracy = accuracy_score(y_test, predictions)
# print(f'Accuracy score with optimizations: {accuracy}')






# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

# # Load your labeled dataset
# df = pd.read_csv('googleplaystore_user_reviews.csv')  # Update the file name to your labeled dataset file

# # Make sure the column names match your dataset structure
# X_train, X_test, y_train, y_test = train_test_split(df['Translated_Review'], df['Sentiment'], test_size=0.2, random_state=42)

# # Create a text processing and classification pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english', max_features=5000),  # Consider adjusting max_features based on your dataset
#     LogisticRegression(random_state=42, C=1, solver='liblinear')  # Adjust hyperparameters as needed
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# # Assuming you have an unlabeled dataset that you want to classify
# unlabeled_df = pd.read_csv('unlabeled_new_dataset.csv')  # Update the file name to your unlabeled dataset file
# unlabeled_texts = unlabeled_df['text']  # Make sure 'text' matches the name of the column containing text in your dataset

# # Predict whether the text is human-generated (1) or AI-generated (0)
# predicted_labels = pipeline.predict(unlabeled_texts)

# # Add the predictions to your unlabeled dataset
# unlabeled_df['predicted_label'] = predicted_labels
# unlabeled_df['predicted_label'] = unlabeled_df['predicted_label'].apply(lambda x: 'Human-Generated' if x == 1 else 'AI-Generated')

# # Save or further process your now labeled dataset
# unlabeled_df.to_csv('classified_new_dataset.csv', index=False)



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

# # Load your labeled dataset
# df = pd.read_csv('app_details.csv')  # Ensure this is your correct file name

# # Handling NaN values by filling them with an empty string
# df['reviews'].fillna('', inplace=True)

# # Assuming 'saleTime' is a numeric or categorical label for classification
# # Ensure 'saleTime' is correctly named and formatted as your target variable

# # Split the dataset into training and testing sets
# # Make sure 'reviews' and 'saleTime' match your dataset's column names
# X_train, X_test, y_train, y_test = train_test_split(df['reviews'], df['saleTime'], test_size=0.2, random_state=42)

# # Create a text processing and classification pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english'),
#     LogisticRegression(random_state=42)
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

# # Load your labeled dataset
# df = pd.read_csv('googleplaystore_user_reviews.csv')  # Update the file name to your labeled dataset file

# # Replace NaN values in the 'Translated_Review' column with an empty string
# df['Translated_Review'].fillna('', inplace=True)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['Translated_Review'], df['Sentiment'], test_size=0.2, random_state=42)

# # Create a text processing and classification pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english'),
#     LogisticRegression(random_state=42)
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# # Assuming you have an unlabeled dataset that you want to classify (optional part)
# # You should also ensure that any 'NaN' values in the unlabeled dataset are handled similarly
# unlabeled_df = pd.read_csv('googleplaystore_user_reviews.csv')  # Update the file name to your unlabeled dataset file
# unlabeled_df['Translated_Review'].fillna('', inplace=True)  # Replace NaN values

# # Predict the labels for the unlabeled dataset
# predicted_labels = pipeline.predict(unlabeled_df['Translated_Review'])

# # Add the predictions to your unlabeled dataset
# unlabeled_df['predicted_label'] = predicted_labels
# # Note: Adjust the label assignment according to your actual labels; the AI-Generated vs. Human-Generated was based on previous context
# # unlabeled_df['predicted_label'] = unlabeled_df['predicted_label'].apply(lambda x: 'AI-Generated' if x == 0 else 'Human-Generated')

# # Save or further process your now labeled dataset
# unlabeled_df.to_csv('classified_dataset.csv', index=False)


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score

# # Load your labeled dataset
# df = pd.read_csv('googleplaystore_user_reviews.csv')  # Update the file name to your labeled dataset file

# # Preprocess: Remove rows where either 'Translated_Review' or 'Sentiment' is NaN
# df.dropna(subset=['Translated_Review', 'Sentiment'], inplace=True)

# # Optionally, replace NaN values in 'Translated_Review' with an empty string
# # This step might be redundant after the dropna but is a good practice if you decide to handle NaNs differently.
# df['Translated_Review'].fillna('', inplace=True)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['Translated_Review'], df['Sentiment'], test_size=0.2, random_state=42)

# # Create a text processing and classification pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english'),
#     LogisticRegression(random_state=42)
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}') ##########################################################







# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score, silhouette_score
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Load the dataset for sentiment analysis
# df_sentiment = pd.read_csv('googleplaystore_user_reviews.csv')

# # Preprocess: Remove rows where either 'Translated_Review' or 'Sentiment' is NaN
# df_sentiment.dropna(subset=['Translated_Review','Sentiment'], inplace=True)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df_sentiment['Translated_Review'], df_sentiment['Sentiment'], test_size=0.2, random_state=42)

# # Create a text processing and classification pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english'),
#     LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter to handle convergence
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print(f'Sentiment Analysis Accuracy: {accuracy_score(y_test, y_pred)}')

# # Load the dataset for clustering
# # Assuming the same dataset is used, we can continue from the preprocessed df_sentiment for a consistent approach
# df_clustering = df_sentiment.copy()

# # Ensure no NaN values in 'Translated_Review' for clustering
# df_clustering['Translated_Review'].fillna('', inplace=True)  # Filling NaN with empty strings

# # Convert texts to TF-IDF features
# vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
# X = vectorizer.fit_transform(df_clustering['Translated_Review'])

# # Preliminary analysis to choose an optimal number of clusters (optional)
# silhouette_scores = []
# for n_clusters in range(2, 7):  # Example: testing between 2 to 6 clusters
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     clusters = kmeans.fit_predict(X)
#     score = silhouette_score(X, clusters)
#     silhouette_scores.append(score)
#     print(f"Silhouette Score for {n_clusters} clusters: {score}")

# # Plot silhouette scores (optional)
# plt.plot(range(2, 7), silhouette_scores, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette Score')
# plt.title('Silhouette Score Analysis')
# plt.show()

# # Assuming you decide on an optimal number of clusters based on the analysis
# optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Automatically selecting the best number of clusters

# # Clustering with the chosen number of clusters
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
# clusters = kmeans.fit_predict(X)
# df_clustering['cluster'] = clusters

# # Inspect texts in each cluster to infer AI-generated vs. human-generated
# for i in range(optimal_clusters):
#     print(f"\nCluster {i} samples:")
#     print(df_clustering[df_clustering['cluster'] == i]['Translated_Review'].head(100).to_string(index=False), '\n')

# # After inspection, if you identify which cluster might correspond to AI or human
# # Update the labeling accordingly - this is speculative
# df_clustering['predicted_label'] = df_clustering['cluster'].apply(lambda x: 'AI-Generated' if x == 0 else 'Human-Generated')

# # Save the updated dataset with clusters
# df_clustering.to_csv('classified_reviews.csv', index=False)



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


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import accuracy_score, silhouette_score
# from sklearn.cluster import KMeans
# from textblob import TextBlob
# import matplotlib.pyplot as plt

# # PARAMETERS
# dataset_path = 'DatingAppReviewsDataset.csv'
# text_column_name = 'Name'  # Adjusted to the correct column names
# label_column_name = 'Review'

# # Load the dataset
# df = pd.read_csv(dataset_path)

# # Preprocess: Remove NaNs
# df.dropna(subset=[text_column_name, label_column_name], inplace=True)

# # Convert text column to string
# df[text_column_name] = df[text_column_name].astype(str)

# # Sentiment Analysis with TextBlob
# def get_sentiment(text):
#     blob = TextBlob(text)
#     return blob.sentiment.polarity

# df['sentiment'] = df[text_column_name].apply(get_sentiment)

# # Split dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df[text_column_name], df[label_column_name], test_size=0.2, random_state=42)

# # Text Processing and Classification Pipeline
# pipeline = make_pipeline(
#     TfidfVectorizer(stop_words='english', max_features=1000),  # Consider lowering max_features if still facing memory issues
#     LogisticRegression(random_state=42, max_iter=100)  # Lowered max_iter for potential performance improvement
# )

# # Train the model
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# y_pred = pipeline.predict(X_test)
# print(f'Classification Accuracy: {accuracy_score(y_test, y_pred)}')

# # Clustering and the rest of your analysis code follows here...
# # Ensure you're working with manageable data sizes and adjust parameters to fit your system's capabilities





