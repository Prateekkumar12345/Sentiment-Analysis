import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from tqdm import tqdm

nltk.download('vader_lexicon')

# Load your dataset
data = "Classified_reviews_with_Sentiment.csv"
df = pd.read_csv(data)

# Splitting the dataset based on the type of generation
human_generated = df[df['predicted_label'] == 'Human Generated']
ai_generated = df[df['predicted_label'] == 'AI Generated'].copy()  # Use .copy() to modify DataFrame safely

print("Human Generated Data:")
print(human_generated.head(20))  # Displaying the first few rows for inspection

print("\nAI Generated Data:")
print(ai_generated.head(20))  # Displaying the first few rows for inspection

# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Define a function to get polarity scores
def get_polarity_scores(text):
    return sia.polarity_scores(text)

# Adding polarity scores to the AI Generated DataFrame
ai_generated['polarity_scores'] = [get_polarity_scores(text) for text in tqdm(ai_generated['Text'])]

# Extracting the 'compound' score to use as the sentiment score
ai_generated['sentiment_score'] = ai_generated['polarity_scores'].apply(lambda x: x['compound'])

# Displaying the updated DataFrame with the sentiment scores
print("\nUpdated AI Generated Data with Sentiment Scores:")
print(ai_generated[['Text', 'polarity_scores', 'sentiment_score']].head(20))

