import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import Counter
import re
from typing import List, Dict, Tuple
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ThematicAnalyzer:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define banking-specific themes and keywords
        self.theme_keywords = {
            'login_issues': [
                'login', 'password', 'forgot', 'reset', 'authentication',
                'biometric', 'fingerprint', 'face id', 'cannot login',
                'access denied', 'locked out'
            ],
            'transaction_problems': [
                'transfer', 'transaction', 'failed', 'pending', 'slow',
                'timeout', 'error', 'declined', 'payment', 'send money',
                'receive money', 'instant', 'delayed'
            ],
            'app_performance': [
                'crash', 'freeze', 'lag', 'slow', 'loading',
                'bug', 'glitch', 'error', 'not working', 'close',
                'force close', 'unresponsive'
            ],
            'user_interface': [
                'ui', 'ux', 'design', 'interface', 'layout',
                'navigation', 'menu', 'button', 'screen', 'display',
                'color', 'font', 'size'
            ],
            'customer_support': [
                'support', 'help', 'service', 'assistance', 'contact',
                'response', 'wait', 'hours', 'email', 'phone',
                'chat', 'complaint', 'resolve'
            ],
            'security_concerns': [
                'security', 'safe', 'hack', 'fraud', 'scam',
                'privacy', 'data', 'information', 'protection',
                'trust', 'secure'
            ],
            'feature_requests': [
                'feature', 'add', 'implement', 'should have',
                'missing', 'need', 'want', 'request', 'suggestion',
                'improvement', 'update', 'new'
            ],
            'account_management': [
                'account', 'balance', 'statement', 'history',
                'details', 'update', 'profile', 'information',
                'settings', 'preferences'
            ]
        }
        
        # Theme descriptions for reporting
        self.theme_descriptions = {
            'login_issues': 'Problems with authentication, password reset, biometric login',
            'transaction_problems': 'Issues with transfers, payments, transaction speed and failures',
            'app_performance': 'App crashes, freezes, lagging, and general performance issues',
            'user_interface': 'Feedback on app design, layout, navigation, and user experience',
            'customer_support': 'Comments about support quality, response time, and service',
            'security_concerns': 'Worries about app security, fraud protection, and data privacy',
            'feature_requests': 'Suggestions for new features or improvements to existing ones',
            'account_management': 'Issues related to account information, balance, and settings'
        }
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text using spaCy"""
        if not text or not isinstance(text, str):
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract nouns, verbs, adjectives
        keywords = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                # Lemmatize
                lemma = token.lemma_.lower()
                if len(lemma) > 2:  # Filter out very short lemmas
                    keywords.append(lemma)
        
        # Count frequencies
        keyword_counts = Counter(keywords)
        
        # Return top N keywords
        return [kw for kw, count in keyword_counts.most_common(top_n)]
    
    def match_to_themes(self, keywords: List[str]) -> Dict[str, float]:
        """Match keywords to predefined themes"""
        theme_scores = {theme: 0 for theme in self.theme_keywords}
        
        if not keywords:
            return theme_scores
        
        # Calculate score for each theme
        for theme, theme_words in self.theme_keywords.items():
            matches = sum(1 for keyword in keywords if any(theme_word in keyword for theme_word in theme_words))
            theme_scores[theme] = matches / len(keywords) if keywords else 0
        
        return theme_scores
    
    def extract_n_grams(self, texts: List[str], n_range: Tuple[int, int] = (2, 3)) -> List[str]:
        """Extract n-grams from list of texts"""
        vectorizer = TfidfVectorizer(
            ngram_range=n_range,
            stop_words='english',
            max_features=50
        )
        
        try:
            X = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            tfidf_scores = X.sum(axis=0).A1
            feature_score_pairs = list(zip(feature_names, tfidf_scores))
            
            # Sort by score
            feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return [feature for feature, score in feature_score_pairs[:20]]
        except Exception as e:
            logger.error(f"Error extracting n-grams: {str(e)}")
            return []
    
    def perform_topic_modeling(self, texts: List[str], n_topics: int = 5) -> Dict:
        """Perform topic modeling using LDA"""
        try:
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_df=0.95,
                min_df=2,
                stop_words='english',
                max_features=1000
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            lda.fit(tfidf_matrix)
            
            # Extract topics
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                topic_words = ', '.join(top_features)
                topics[f'topic_{topic_idx + 1}'] = {
                    'words': topic_words,
                    'keywords': top_features
                }
            
            return topics
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            return {}
    
    def analyze_themes_by_bank(self, df: pd.DataFrame) -> Dict:
        """Analyze themes for each bank"""
        results = {}
        
        for bank in df['bank'].unique():
            bank_df = df[df['bank'] == bank]
            
            # Combine all reviews for the bank
            all_texts = bank_df['cleaned_text'].tolist()
            
            # Extract common n-grams
            common_ngrams = self.extract_n_grams(all_texts)
            
            # Perform topic modeling
            topics = self.perform_topic_modeling(all_texts)
            
            # Analyze sentiment distribution within themes
            theme_analysis = {}
            
            for theme_name in self.theme_keywords:
                # Find reviews matching this theme
                theme_reviews = []
                for idx, row in bank_df.iterrows():
                    keywords = self.extract_keywords(row['cleaned_text'])
                    theme_scores = self.match_to_themes(keywords)
                    if theme_scores[theme_name] > 0.3:  # Threshold
                        theme_reviews.append(row)
                
                if theme_reviews:
                    theme_df = pd.DataFrame(theme_reviews)
                    theme_analysis[theme_name] = {
                        'count': len(theme_reviews),
                        'percentage': (len(theme_reviews) / len(bank_df)) * 100,
                        'avg_rating': theme_df['rating'].mean() if 'rating' in theme_df.columns else None,
                        'sentiment_distribution': {
                            'positive': len(theme_df[theme_df['sentiment_label'] == 'POSITIVE']) if 'sentiment_label' in theme_df.columns else 0,
                            'negative': len(theme_df[theme_df['sentiment_label'] == 'NEGATIVE']) if 'sentiment_label' in theme_df.columns else 0,
                            'neutral': len(theme_df[theme_df['sentiment_label'] == 'NEUTRAL']) if 'sentiment_label' in theme_df.columns else 0,
                        },
                        'top_keywords': self.extract_keywords(' '.join(theme_df['cleaned_text'].tolist()), top_n=10),
                        'example_reviews': theme_df['review_text'].head(3).tolist() if 'review_text' in theme_df.columns else []
                    }
            
            results[bank] = {
                'total_reviews': len(bank_df),
                'common_ngrams': common_ngrams,
                'topics': topics,
                'theme_analysis': theme_analysis,
                'most_common_themes': sorted(
                    [(theme, data['count']) for theme, data in theme_analysis.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
        
        return results
    
    def generate_wordcloud(self, texts: List[str], bank_name: str, save_path: str = None):
        """Generate word cloud for bank reviews"""
        # Combine all texts
        all_text = ' '.join(texts)
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50,
            contour_width=3,
            contour_color='steelblue'
        ).generate(all_text)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Most Common Words - {bank_name}', fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_thematic_analysis(self, results: Dict, output_file: str = 'data/processed/thematic_analysis.json'):
        """Save thematic analysis results"""
        import json
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                return obj
            
            serializable_results = json.loads(json.dumps(results, default=convert_to_serializable))
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Thematic analysis saved to: {output_file}")

if __name__ == "__main__":
    # Load data with sentiment analysis
    df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
    
    # Initialize analyzer
    analyzer = ThematicAnalyzer()
    
    # Analyze themes by bank
    print("Analyzing themes for each bank...")
    results = analyzer.analyze_themes_by_bank(df)
    
    # Save results
    analyzer.save_thematic_analysis(results)
    
    # Generate word clouds
    print("\nGenerating word clouds...")
    for bank in df['bank'].unique():
        bank_texts = df[df['bank'] == bank]['cleaned_text'].tolist()
        save_path = f'data/processed/wordcloud_{bank}.png'
        analyzer.generate_wordcloud(bank_texts, bank, save_path)
        print(f"✓ Word cloud saved for {bank}: {save_path}")
    
    # Print summary
    print("\n=== Thematic Analysis Summary ===")
    for bank, bank_results in results.items():
        print(f"\n{bank.upper()}:")
        print(f"  Total Reviews: {bank_results['total_reviews']}")
        print(f"  Most Common Themes:")
        for theme, count in bank_results['most_common_themes']:
            percentage = (count / bank_results['total_reviews']) * 100
            print(f"    - {theme}: {count} reviews ({percentage:.1f}%)")
        
        print(f"\n  Top 5 Common Phrases:")
        for i, phrase in enumerate(bank_results['common_ngrams'][:5], 1):
            print(f"    {i}. {phrase}")