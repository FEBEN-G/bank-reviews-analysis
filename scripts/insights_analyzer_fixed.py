"""
Simplified Insights Analyzer for Task 4
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class InsightsAnalyzerFixed:
    def __init__(self, data_path='data/processed/combined_reviews.csv'):
        self.data_path = data_path
        self.df = None
        self.banks = ['CBE', 'BOA', 'Dashen']
        
    def load_data(self):
        """Load and prepare data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} reviews")
            
            # Ensure required columns exist
            self._ensure_columns()
            
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _ensure_columns(self):
        """Ensure all required columns exist"""
        # Create primary_theme if it doesn't exist
        if 'primary_theme' not in self.df.columns:
            print("⚠️  Creating primary_theme column...")
            self.df['primary_theme'] = 'General Feedback'
            
            # Try to create from theme columns
            theme_cols = [col for col in self.df.columns if col.startswith('theme_')]
            if theme_cols:
                def get_theme(row):
                    for col in theme_cols:
                        if row[col] == 1:
                            return col.replace('theme_', '').replace('_', ' ').title()
                    return 'General Feedback'
                
                self.df['primary_theme'] = self.df.apply(get_theme, axis=1)
        
        # Create cleaned text if needed
        if 'review_text_clean' not in self.df.columns:
            text_col = 'review_text' if 'review_text' in self.df.columns else 'review'
            if text_col in self.df.columns:
                self.df['review_text_clean'] = self.df[text_col].astype(str).str.lower().str.replace('[^\w\s]', ' ', regex=True)
        
        # Ensure sentiment_label is consistent
        if 'sentiment_label' in self.df.columns:
            self.df['sentiment_label'] = self.df['sentiment_label'].astype(str).str.upper()
    
    def calculate_metrics(self):
        """Calculate basic metrics"""
        metrics = {}
        for bank in self.banks:
            bank_df = self.df[self.df['bank'] == bank]
            
            metrics[bank] = {
                'total_reviews': len(bank_df),
                'avg_rating': bank_df['rating'].mean() if 'rating' in bank_df.columns else 3.0,
                'avg_sentiment': bank_df['sentiment_score'].mean() if 'sentiment_score' in bank_df.columns else 0.0,
                'positive_pct': self._get_sentiment_pct(bank_df, 'POSITIVE'),
                'negative_pct': self._get_sentiment_pct(bank_df, 'NEGATIVE'),
                'neutral_pct': self._get_sentiment_pct(bank_df, 'NEUTRAL'),
                'top_themes': self._get_top_themes(bank_df)
            }
        
        return metrics
    
    def _get_sentiment_pct(self, df, sentiment):
        """Get percentage of specific sentiment"""
        if 'sentiment_label' not in df.columns:
            return 33.3
        
        total = len(df)
        if total == 0:
            return 0
        
        count = len(df[df['sentiment_label'].str.contains(sentiment, case=False, na=False)])
        return round((count / total) * 100, 1)
    
    def _get_top_themes(self, df):
        """Get top themes"""
        if 'primary_theme' not in df.columns:
            return ['General Feedback']
        
        return df['primary_theme'].value_counts().head(3).index.tolist()
    
    def analyze_drivers_pain_points(self):
        """Analyze drivers and pain points"""
        insights = {}
        
        for bank in self.banks:
            bank_df = self.df[self.df['bank'] == bank]
            
            # Simple keyword extraction
            positive_keywords = self._extract_keywords(
                bank_df[bank_df['sentiment_label'].str.contains('POSITIVE', case=False, na=False)], 
                top_n=5
            )
            
            negative_keywords = self._extract_keywords(
                bank_df[bank_df['sentiment_label'].str.contains('NEGATIVE', case=False, na=False)],
                top_n=5
            )
            
            insights[bank] = {
                'drivers': positive_keywords,
                'pain_points': negative_keywords,
                'top_themes': self._get_top_themes(bank_df)
            }
        
        return insights
    
    def _extract_keywords(self, df, top_n=5):
        """Simple keyword extraction"""
        if len(df) == 0 or 'review_text_clean' not in df.columns:
            return ['good', 'excellent', 'helpful'][:top_n]
        
        text = ' '.join(df['review_text_clean'].fillna('').astype(str))
        words = text.split()
        
        # Common words to exclude
        stop_words = {'the', 'and', 'for', 'this', 'that', 'with', 'have', 'has', 'was', 'were', 
                     'are', 'is', 'in', 'on', 'at', 'to', 'of', 'a', 'an', 'my', 'i', 'it'}
        
        # Count word frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Remove stop words and short words
        keywords = [(word, count) for word, count in word_counts.items() 
                   if word not in stop_words and len(word) > 3]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, count in keywords[:top_n]]
    
    def generate_recommendations(self, insights):
        """Generate recommendations"""
        recommendations = {}
        
        for bank in self.banks:
            recs = []
            bank_insights = insights[bank]
            
            # Based on pain points
            for pain in bank_insights['pain_points'][:3]:
                pain_lower = str(pain).lower()
                if any(word in pain_lower for word in ['slow', 'loading', 'lag']):
                    recs.append("Optimize app performance and loading times")
                elif any(word in pain_lower for word in ['crash', 'error', 'bug']):
                    recs.append("Improve app stability and fix bugs")
                elif any(word in pain_lower for word in ['login', 'password']):
                    recs.append("Enhance authentication system")
                else:
                    recs.append(f"Address issues related to '{pain}'")
            
            # Based on themes
            for theme in bank_insights['top_themes']:
                theme_lower = str(theme).lower()
                if 'ui' in theme_lower or 'interface' in theme_lower:
                    recs.append("Redesign user interface for better experience")
                elif 'performance' in theme_lower:
                    recs.append("Conduct performance optimization")
                elif 'support' in theme_lower:
                    recs.append("Improve customer support response")
            
            # Ensure at least 3 recommendations
            while len(recs) < 3:
                recs.append("Monitor user feedback and address common issues")
            
            recommendations[bank] = recs[:3]
        
        return recommendations
    
    def save_report(self, metrics, insights, recommendations):
        """Save report"""
        report = {
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_reviews': len(self.df),
            'banks': self.banks,
            'metrics': metrics,
            'insights': insights,
            'recommendations': recommendations
        }
        
        with open('reports/task4_simple_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✅ Report saved to reports/task4_simple_report.json")
        return report
