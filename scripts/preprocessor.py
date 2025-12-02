import pandas as pd
import numpy as np
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import logging

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class ReviewPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        
    def load_data(self, filepath):
        """Load raw review data"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} reviews from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove special characters and numbers (keep only letters and basic punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize_text(self, text):
        """Tokenize text and remove stopwords"""
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and word not in self.punctuation]
        return tokens
    
    def preprocess_dataframe(self, df):
        """Apply preprocessing to entire dataframe"""
        logger.info("Starting preprocessing...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # 1. Remove duplicates
        initial_count = len(df_processed)
        df_processed = df_processed.drop_duplicates(subset=['review_id'], keep='first')
        logger.info(f"Removed {initial_count - len(df_processed)} duplicate reviews")
        
        # 2. Handle missing values
        missing_before = df_processed.isnull().sum().sum()
        
        # Drop rows with missing review text or rating
        df_processed = df_processed.dropna(subset=['review_text', 'rating'])
        
        # Fill other missing values
        df_processed['reviewer_name'] = df_processed['reviewer_name'].fillna('Anonymous')
        df_processed['reply_text'] = df_processed['reply_text'].fillna('')
        df_processed['reply_date'] = df_processed['reply_date'].fillna('')
        
        missing_after = df_processed.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        # 3. Clean text data
        df_processed['cleaned_text'] = df_processed['review_text'].apply(self.clean_text)
        
        # 4. Tokenize text
        df_processed['tokens'] = df_processed['cleaned_text'].apply(self.tokenize_text)
        
        # 5. Create word count column
        df_processed['word_count'] = df_processed['tokens'].apply(len)
        
        # 6. Filter out very short reviews
        df_processed = df_processed[df_processed['word_count'] >= 3]
        
        # 7. Convert date to datetime
        df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')
        
        # 8. Convert rating to integer
        df_processed['rating'] = pd.to_numeric(df_processed['rating'], errors='coerce').astype('Int64')
        
        # 9. Create review category based on rating
        def categorize_rating(rating):
            if rating >= 4:
                return 'positive'
            elif rating == 3:
                return 'neutral'
            else:
                return 'negative'
        
        df_processed['rating_category'] = df_processed['rating'].apply(categorize_rating)
        
        # 10. Extract bank name from app_name
        df_processed['bank'] = df_processed['app_name'].str.extract(r'(CBE|Abyssinia|Dashen)', expand=False)
        
        logger.info(f"Preprocessing complete. Final records: {len(df_processed)}")
        return df_processed
    
    def save_processed_data(self, df, filename='data/processed/reviews_processed.csv'):
        """Save processed data to CSV"""
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Saved {len(df)} processed reviews to {filename}")
        return filename
    
    def generate_summary(self, df):
        """Generate preprocessing summary"""
        summary = {
            'total_reviews': len(df),
            'reviews_per_bank': df['bank'].value_counts().to_dict(),
            'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
            'rating_category_distribution': df['rating_category'].value_counts().to_dict(),
            'average_word_count': df['word_count'].mean(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d')
            },
            'missing_data_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        }
        
        return summary

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ReviewPreprocessor()
    
    # Load raw data
    raw_data = preprocessor.load_data('data/raw/google_play_reviews_raw.csv')
    
    if not raw_data.empty:
        # Preprocess data
        processed_data = preprocessor.preprocess_dataframe(raw_data)
        
        # Save processed data
        preprocessor.save_processed_data(processed_data)
        
        # Generate and print summary
        summary = preprocessor.generate_summary(processed_data)
        
        print("\n=== Preprocessing Summary ===")
        print(f"Total Reviews: {summary['total_reviews']}")
        print(f"\nReviews per Bank:")
        for bank, count in summary['reviews_per_bank'].items():
            print(f"  {bank}: {count}")
        print(f"\nRating Distribution:")
        for rating, count in sorted(summary['rating_distribution'].items()):
            print(f"  {rating} stars: {count}")
        print(f"\nRating Category Distribution:")
        for category, count in summary['rating_category_distribution'].items():
            print(f"  {category}: {count}")
        print(f"\nAverage Word Count: {summary['average_word_count']:.2f}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Missing Data Percentage: {summary['missing_data_percentage']:.2f}%")