"""
Insights and Recommendations Analyzer
Task 4: Generate insights, visualizations, and recommendations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class InsightsAnalyzer:
    def __init__(self, data_path='data/processed/combined_reviews.csv'):
        """Initialize insights analyzer"""
        self.data_path = data_path
        self.df = None
        self.banks = ['CBE', 'BOA', 'Dashen']
        
    def load_data(self):
        """Load processed review data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.df)} reviews")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def calculate_key_metrics(self):
        """Calculate key metrics for each bank"""
        metrics = {}
        
        for bank in self.banks:
            bank_df = self.df[self.df['bank'] == bank]
            
            metrics[bank] = {
                'total_reviews': len(bank_df),
                'avg_rating': bank_df['rating'].mean(),
                'avg_sentiment': bank_df['sentiment_score'].mean(),
                'positive_pct': len(bank_df[bank_df['sentiment_label'] == 'positive']) / len(bank_df) * 100,
                'negative_pct': len(bank_df[bank_df['sentiment_label'] == 'negative']) / len(bank_df) * 100,
                'neutral_pct': len(bank_df[bank_df['sentiment_label'] == 'neutral']) / len(bank_df) * 100,
                'common_themes': bank_df['primary_theme'].value_counts().head(5).to_dict()
            }
        
        return metrics
    
    def identify_drivers_pain_points(self):
        """Identify satisfaction drivers and pain points"""
        insights = {}
        
        for bank in self.banks:
            bank_df = self.df[self.df['bank'] == bank]
            
            # Drivers (positive reviews with high ratings)
            positive_reviews = bank_df[(bank_df['sentiment_label'] == 'positive') & (bank_df['rating'] >= 4)]
            drivers = self._extract_keywords(positive_reviews, 'review_text_clean', top_n=10)
            
            # Pain points (negative reviews with low ratings)
            negative_reviews = bank_df[(bank_df['sentiment_label'] == 'negative') & (bank_df['rating'] <= 2)]
            pain_points = self._extract_keywords(negative_reviews, 'review_text_clean', top_n=10)
            
            insights[bank] = {
                'drivers': drivers[:5],  # Top 5 drivers
                'pain_points': pain_points[:5],  # Top 5 pain points
                'top_themes': bank_df['primary_theme'].value_counts().head(3).index.tolist()
            }
        
        return insights
    
    def _extract_keywords(self, df, column, top_n=10):
        """Extract keywords from text column"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        if len(df) == 0:
            return []
        
        vectorizer = CountVectorizer(stop_words='english', max_features=top_n*2)
        X = vectorizer.fit_transform(df[column].fillna(''))
        
        # Get feature names and counts
        word_counts = X.sum(axis=0).A1
        feature_names = vectorizer.get_feature_names_out()
        
        # Sort by frequency
        word_freq = list(zip(feature_names, word_counts))
        word_freq.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in word_freq[:top_n]]
    
    def generate_recommendations(self, insights):
        """Generate actionable recommendations for each bank"""
        recommendations = {}
        
        for bank in self.banks:
            bank_insights = insights[bank]
            
            recs = []
            
            # Based on pain points
            for pain in bank_insights['pain_points'][:3]:
                if 'slow' in pain or 'loading' in pain:
                    recs.append(f"Optimize app performance and reduce loading times for {pain}-related operations")
                elif 'crash' in pain or 'error' in pain:
                    recs.append(f"Improve app stability and fix {pain} bugs through rigorous testing")
                elif 'login' in pain or 'password' in pain:
                    recs.append(f"Enhance authentication system and simplify {pain} process")
                elif 'transfer' in pain or 'transaction' in pain:
                    recs.append(f"Streamline {pain} process and add transaction confirmation steps")
            
            # Based on themes
            for theme in bank_insights['top_themes']:
                if 'UI' in theme:
                    recs.append("Redesign user interface for better navigation and modern look")
                elif 'Customer Support' in theme:
                    recs.append("Implement AI chatbot for 24/7 customer support")
                elif 'Performance' in theme:
                    recs.append("Conduct performance audit and optimize resource usage")
                elif 'Features' in theme:
                    recs.append("Prioritize most requested features in next update")
            
            recommendations[bank] = recs[:3]  # Top 3 recommendations
        
        return recommendations
    
    def save_insights_report(self, metrics, insights, recommendations):
        """Save comprehensive insights report"""
        report = {
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'project': 'Bank Reviews Analysis - Task 4',
            'total_reviews_analyzed': len(self.df),
            'banks_analyzed': self.banks,
            'key_metrics': metrics,
            'insights': insights,
            'recommendations': recommendations,
            'scenario_analysis': self._analyze_scenarios(metrics, insights)
        }
        
        # Save JSON report
        report_path = 'reports/task4_insights_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Insights report saved to {report_path}")
        return report
    
    def _analyze_scenarios(self, metrics, insights):
        """Analyze assignment scenarios"""
        scenarios = {
            'scenario_1_retaining_users': {
                'description': 'CBE has 4.2, BOA 3.4, and Dashen 4.1 star rating. Users complain about slow loading during transfers.',
                'analysis': self._analyze_scenario_1(metrics, insights),
                'recommendations': [
                    'Implement performance monitoring for transfer operations',
                    'Optimize database queries and cache frequently accessed data',
                    'Add progress indicators during transfers'
                ]
            },
            'scenario_2_enhancing_features': {
                'description': 'Extract desired features through keyword analysis.',
                'analysis': self._analyze_scenario_2(insights),
                'recommendations': [
                    'Prioritize biometric authentication features',
                    'Add budgeting and financial planning tools',
                    'Implement real-time notifications for transactions'
                ]
            },
            'scenario_3_managing_complaints': {
                'description': 'Cluster and track complaints for AI chatbot integration.',
                'analysis': self._analyze_scenario_3(insights),
                'recommendations': [
                    'Train AI chatbot on common complaint patterns',
                    'Implement automated ticket categorization',
                    'Create knowledge base for frequent issues'
                ]
            }
        }
        return scenarios
    
    def _analyze_scenario_1(self, metrics, insights):
        """Analyze user retention scenario"""
        analysis = []
        
        # Check if slow loading is a common issue
        for bank in self.banks:
            if any('slow' in str(point).lower() or 'loading' in str(point).lower() 
                   for point in insights[bank]['pain_points']):
                analysis.append(f"{bank}: Slow loading confirmed as pain point")
            else:
                analysis.append(f"{bank}: Slow loading not among top pain points")
        
        # Compare ratings
        rating_comparison = {bank: metrics[bank]['avg_rating'] for bank in self.banks}
        analysis.append(f"Rating comparison: {rating_comparison}")
        
        return analysis
    
    def _analyze_scenario_2(self, insights):
        """Analyze feature enhancement scenario"""
        analysis = []
        
        for bank in self.banks:
            desired_features = [point for point in insights[bank]['pain_points'] 
                              if 'feature' in str(point).lower() or 'add' in str(point).lower()]
            analysis.append(f"{bank}: Desired features - {desired_features[:3]}")
        
        return analysis
    
    def _analyze_scenario_3(self, insights):
        """Analyze complaint management scenario"""
        analysis = []
        
        for bank in self.banks:
            complaints = insights[bank]['pain_points']
            analysis.append(f"{bank}: Top complaints for chatbot - {complaints[:3]}")
        
        return analysis
