# Dating App Reviews Analysis

This project analyzes reviews from a dating app dataset using various natural language processing (NLP) and machine learning techniques. The analysis includes sentiment analysis, text classification, and clustering.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)
- [License](#license)

## Dataset

The dataset should be a CSV file containing at least the following columns:
- `Review`: The text of the review.
- `Rating`: The rating given in the review.

## Requirements

- pandas
- scikit-learn
- textblob
- matplotlib

You can install the necessary Python packages using pip:


pip install pandas scikit-learn textblob matplotlib


Script Details
The script performs the following steps:

Load the Dataset: Reads the dataset from a CSV file.
Preprocess Data: Removes NaN values and converts the text column to strings.
Sentiment Analysis: Uses TextBlob to calculate the sentiment polarity of each review.
Text Classification: Splits the data into training and testing sets, then trains a Logistic Regression model using TF-IDF features.
Model Evaluation: Prints the classification accuracy of the model.
Clustering: Uses KMeans clustering to group the reviews and performs silhouette analysis to find the optimal number of clusters.
Assign Clusters: Adds the cluster labels to the dataframe.
Speculative Labeling: Assigns speculative labels based on cluster analysis.
Save Results: Saves the updated dataset with sentiment scores, cluster labels, and speculative labels to a new CSV file.
Output
The script generates the following output files:

classified_reviews_with_sentiment.csv: The original dataset with added sentiment scores, cluster labels, and speculative labels.
