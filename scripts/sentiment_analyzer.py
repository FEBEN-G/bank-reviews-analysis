import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import torch
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        """Initialize sentiment analyzer with multiple models"""
        self.model_name = model_name
        
        # Initialize VADER
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Transformers model
        try:
            self.transformers_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded transformers model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformers model: {str(e)}")
            self.transformers_pipeline = None
    
    def analyze_vader(self, text: str) -> Dict:
        """Analyze sentiment using VADER"""
        if not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.0, 'compound': 0.0}
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine label based on compound score
        compound = scores['compound']
        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': abs(compound),
            'compound': compound,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_transformers(self, text: str) -> Dict:
        """Analyze sentiment using transformers model"""
        if not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        if self.transformers_pipeline is None:
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.transformers_pipeline(text)[0]
            
            # Convert to consistent format
            label = result['label'].upper()
            score = result['score']
            
            return {'label': label, 'score': score}
        except Exception as e:
            logger.error(f"Transformers analysis error: {str(e)}")
            return {'label': 'NEUTRAL', 'score': 0.0}
    
    def analyze_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        if not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.0, 'polarity': 0.0, 'subjectivity': 0.0}
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine label
        if polarity > 0:
            label = 'POSITIVE'
        elif polarity < 0:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': abs(polarity),
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def analyze_sentiment(self, text: str, method: str = 'ensemble') -> Dict:
        """Analyze sentiment using specified method"""
        if method == 'vader':
            return self.analyze_vader(text)
        elif method == 'transformers':
            return self.analyze_transformers(text)
        elif method == 'textblob':
            return self.analyze_textblob(text)
        elif method == 'ensemble':
            # Ensemble method - combine all three
            vader_result = self.analyze_vader(text)
            transformers_result = self.analyze_transformers(text)
            textblob_result = self.analyze_textblob(text)
            
            # Weighted voting (adjust weights as needed)
            weights = {'vader': 0.4, 'transformers': 0.4, 'textblob': 0.2}
            
            # Collect votes
            votes = {
                vader_result['label']: weights['vader'],
                transformers_result['label']: weights['transformers'],
                textblob_result['label']: weights['textblob']
            }
            
            # Determine final label
            final_label = max(votes, key=votes.get)
            
            # Calculate average score
            scores = [
                vader_result['score'] * weights['vader'],
                transformers_result['score'] * weights['transformers'],
                textblob_result['score'] * weights['textblob']
            ]
            avg_score = sum(scores)
            
            return {
                'label': final_label,
                'score': avg_score,
                'vader_label': vader_result['label'],
                'vader_score': vader_result['score'],
                'transformers_label': transformers_result['label'],
                'transformers_score': transformers_result['score'],
                'textblob_label': textblob_result['label'],
                'textblob_score': textblob_result['score']
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """Analyze sentiment for entire dataframe"""
        logger.info("Starting sentiment analysis...")
        
        # Create a copy
        df_analyzed = df.copy()
        
        # Initialize results lists
        sentiment_labels = []
        sentiment_scores = []
        sentiment_details = []
        
        # Process each review
        for idx, row in df_analyzed.iterrows():
            text = row[text_column]
            
            # Use ensemble method
            result = self.analyze_sentiment(text, method='ensemble')
            
            sentiment_labels.append(result['label'])
            sentiment_scores.append(result['score'])
            sentiment_details.append(result)
            
            # Log progress
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df_analyzed)} reviews")
        
        # Add results to dataframe
        df_analyzed['sentiment_label'] = sentiment_labels
        df_analyzed['sentiment_score'] = sentiment_scores
        df_analyzed['sentiment_details'] = sentiment_details
        
        # Extract individual model results for analysis
        df_analyzed['vader_label'] = df_analyzed['sentiment_details'].apply(lambda x: x.get('vader_label', 'NEUTRAL'))
        df_analyzed['transformers_label'] = df_analyzed['sentiment_details'].apply(lambda x: x.get('transformers_label', 'NEUTRAL'))
        df_analyzed['textblob_label'] = df_analyzed['sentiment_details'].apply(lambda x: x.get('textblob_label', 'NEUTRAL'))
        
        logger.info(f"Sentiment analysis complete. Processed {len(df_analyzed)} reviews")
        
        return df_analyzed
    
    def generate_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive sentiment summary"""
        summary = {
            'overall': {
                'positive': len(df[df['sentiment_label'] == 'POSITIVE']),
                'negative': len(df[df['sentiment_label'] == 'NEGATIVE']),
                'neutral': len(df[df['sentiment_label'] == 'NEUTRAL']),
                'total': len(df)
            },
            'by_bank': {},
            'by_rating': {},
            'sentiment_vs_rating': {},
            'model_agreement': {}
        }
        
        # Sentiment by bank
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            summary['by_bank'][bank] = {
                'positive': len(bank_df[bank_df['sentiment_label'] == 'POSITIVE']),
                'negative': len(bank_df[bank_df['sentiment_label'] == 'NEGATIVE']),
                'neutral': len(bank_df[bank_df['sentiment_label'] == 'NEUTRAL']),
                'total': len(bank_df),
                'avg_sentiment_score': bank_df['sentiment_score'].mean()
            }
        
        # Sentiment by rating
        for rating in sorted(df['rating'].unique()):
            rating_df = df[df['rating'] == rating]
            summary['by_rating'][str(rating)] = {
                'positive': len(rating_df[rating_df['sentiment_label'] == 'POSITIVE']),
                'negative': len(rating_df[rating_df['sentiment_label'] == 'NEGATIVE']),
                'neutral': len(rating_df[rating_df['sentiment_label'] == 'NEUTRAL']),
                'total': len(rating_df),
                'avg_sentiment_score': rating_df['sentiment_score'].mean()
            }
        
        # Model agreement analysis
        agreement_count = 0
        for idx, row in df.iterrows():
            labels = [row['vader_label'], row['transformers_label'], row['textblob_label']]
            if len(set(labels)) == 1:  # All models agree
                agreement_count += 1
        
        summary['model_agreement'] = {
            'agreement_count': agreement_count,
            'total_reviews': len(df),
            'agreement_percentage': (agreement_count / len(df)) * 100
        }
        
        # Sentiment vs Rating correlation
        sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
        df['sentiment_numeric'] = df['sentiment_label'].map(sentiment_mapping)
        
        try:
            correlation = df['rating'].corr(df['sentiment_numeric'])
            summary['sentiment_vs_rating'] = {
                'correlation': correlation,
                'interpretation': 'strong positive' if correlation > 0.7 else 
                                 'moderate positive' if correlation > 0.3 else 
                                 'weak positive' if correlation > 0 else 
                                 'negative' if correlation < 0 else 'no correlation'
            }
        except:
            summary['sentiment_vs_rating'] = {'correlation': None, 'interpretation': 'Cannot compute'}
        
        return summary

if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('data/processed/reviews_processed.csv')
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    df_analyzed = analyzer.analyze_dataframe(df)
    
    # Save results
    output_file = 'data/processed/reviews_with_sentiment.csv'
    df_analyzed.to_csv(output_file, index=False)
    print(f"Saved sentiment analysis results to: {output_file}")
    
    # Generate summary
    summary = analyzer.generate_sentiment_summary(df_analyzed)
    
    # Print summary
    print("\n=== Sentiment Analysis Summary ===")
    print(f"\nOverall Sentiment:")
    print(f"  Positive: {summary['overall']['positive']} ({summary['overall']['positive']/summary['overall']['total']*100:.1f}%)")
    print(f"  Negative: {summary['overall']['negative']} ({summary['overall']['negative']/summary['overall']['total']*100:.1f}%)")
    print(f"  Neutral: {summary['overall']['neutral']} ({summary['overall']['neutral']/summary['overall']['total']*100:.1f}%)")
    
    print(f"\nModel Agreement: {summary['model_agreement']['agreement_percentage']:.1f}%")
    print(f"Sentiment-Rating Correlation: {summary['sentiment_vs_rating'].get('correlation', 0):.2f} ({summary['sentiment_vs_rating'].get('interpretation', 'N/A')})")