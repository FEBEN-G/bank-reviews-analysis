"""
Visualization Generator for Task 4
Create plots for final report
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class VisualizationGenerator:
    def __init__(self, data_path='data/processed/combined_reviews.csv'):
        self.data_path = data_path
        self.df = None
        self.banks = ['CBE', 'BOA', 'Dashen']
        self.colors = {'CBE': '#2E86AB', 'BOA': '#A23B72', 'Dashen': '#F18F01'}
        
    def load_data(self):
        """Load review data"""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} reviews for visualization")
        return self.df
    
    def plot_rating_distribution(self, save_path='reports/figures/rating_distribution.png'):
        """Plot rating distribution by bank"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Rating Distribution by Bank', fontsize=16, fontweight='bold')
        
        for idx, bank in enumerate(self.banks):
            ax = axes[idx]
            bank_data = self.df[self.df['bank'] == bank]
            
            rating_counts = bank_data['rating'].value_counts().sort_index()
            colors = ['#FF6B6B', '#FFD166', '#FFD166', '#06D6A0', '#06D6A0']
            
            bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, 
                         color=colors, edgecolor='black', alpha=0.8)
            
            ax.set_title(f'{bank}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Rating (Stars)', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved rating distribution to {save_path}")
        return save_path
    
    def plot_sentiment_comparison(self, save_path='reports/figures/sentiment_comparison.png'):
        """Plot sentiment comparison across banks"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Sentiment distribution
        sentiment_data = []
        for bank in self.banks:
            bank_df = self.df[self.df['bank'] == bank]
            for sentiment in ['positive', 'neutral', 'negative']:
                count = len(bank_df[bank_df['sentiment_label'] == sentiment])
                sentiment_data.append({'bank': bank, 'sentiment': sentiment, 'count': count})
        
        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_pivot = sentiment_df.pivot(index='bank', columns='sentiment', values='count')
        
        sentiment_pivot.plot(kind='bar', stacked=True, ax=ax1, 
                           color=['#06D6A0', '#FFD166', '#EF476F'], edgecolor='black')
        ax1.set_title('Sentiment Distribution by Bank', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Bank', fontsize=12)
        ax1.set_ylabel('Number of Reviews', fontsize=12)
        ax1.legend(title='Sentiment')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Average sentiment score
        avg_sentiment = self.df.groupby('bank')['sentiment_score'].mean().reindex(self.banks)
        bars = ax2.bar(avg_sentiment.index, avg_sentiment.values, 
                      color=[self.colors[b] for b in self.banks], edgecolor='black', alpha=0.8)
        ax2.set_title('Average Sentiment Score by Bank', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bank', fontsize=12)
        ax2.set_ylabel('Average Sentiment Score', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved sentiment comparison to {save_path}")
        return save_path
    
    def plot_theme_analysis(self, save_path='reports/figures/theme_analysis.png'):
        """Plot thematic analysis results"""
        # Create theme data
        theme_data = []
        for bank in self.banks:
            bank_df = self.df[self.df['bank'] == bank]
            theme_counts = bank_df['primary_theme'].value_counts().head(5)
            for theme, count in theme_counts.items():
                theme_data.append({'bank': bank, 'theme': theme, 'count': count})
        
        theme_df = pd.DataFrame(theme_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Top Themes by Bank', fontsize=16, fontweight='bold')
        
        for idx, bank in enumerate(self.banks):
            ax = axes[idx]
            bank_themes = theme_df[theme_df['bank'] == bank].sort_values('count', ascending=True)
            
            if len(bank_themes) > 0:
                bars = ax.barh(bank_themes['theme'], bank_themes['count'], 
                             color=self.colors[bank], alpha=0.8, edgecolor='black')
                ax.set_title(f'{bank}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Number of Reviews', fontsize=12)
                
                # Add count labels
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                           f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved theme analysis to {save_path}")
        return save_path
    
    def generate_wordclouds(self):
        """Generate word clouds for each bank"""
        from wordcloud import WordCloud, STOPWORDS
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Word Clouds of Reviews by Bank', fontsize=16, fontweight='bold')
        
        for idx, bank in enumerate(self.banks):
            ax = axes[idx]
            bank_text = ' '.join(self.df[self.df['bank'] == bank]['review_text_clean'].fillna('').astype(str))
            
            if bank_text.strip():
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    stopwords=STOPWORDS,
                    contour_width=1,
                    contour_color='steelblue'
                ).generate(bank_text)
                
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{bank}', fontsize=14, fontweight='bold')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('reports/figures/wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Saved word clouds to reports/figures/wordclouds.png")
        return 'reports/figures/wordclouds.png'
    
    def plot_temporal_trends(self, save_path='reports/figures/temporal_trends.png'):
        """Plot sentiment trends over time"""
        # Convert date column if needed
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
        self.df['month'] = self.df['date'].dt.to_period('M').astype(str)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Monthly review volume
        monthly_counts = self.df.groupby(['month', 'bank']).size().unstack(fill_value=0)
        monthly_counts[self.banks].plot(ax=axes[0], marker='o', linewidth=2, markersize=6)
        axes[0].set_title('Monthly Review Volume by Bank', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Month', fontsize=12)
        axes[0].set_ylabel('Number of Reviews', fontsize=12)
        axes[0].legend(title='Bank')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Monthly average rating
        monthly_rating = self.df.groupby(['month', 'bank'])['rating'].mean().unstack()
        monthly_rating[self.banks].plot(ax=axes[1], marker='s', linewidth=2, markersize=6)
        axes[1].set_title('Monthly Average Rating by Bank', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Average Rating', fontsize=12)
        axes[1].legend(title='Bank')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved temporal trends to {save_path}")
        return save_path
    
    def plot_recommendations_matrix(self, insights_report_path='reports/task4_insights_report.json',
                                   save_path='reports/figures/recommendations_matrix.png'):
        """Plot recommendations matrix"""
        import json
        
        with open(insights_report_path, 'r') as f:
            report = json.load(f)
        
        recommendations = report['recommendations']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Priority Recommendations by Bank', fontsize=16, fontweight='bold')
        
        for idx, bank in enumerate(self.banks):
            ax = axes[idx]
            bank_recs = recommendations.get(bank, [])
            
            if bank_recs:
                # Create horizontal bar chart
                y_pos = np.arange(len(bank_recs))
                colors = plt.cm.RdYlGn(np.linspace(0.3, 0.7, len(bank_recs)))
                
                bars = ax.barh(y_pos, [1] * len(bank_recs), color=colors, edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels([f'Rec {i+1}' for i in range(len(bank_recs))])
                ax.set_xlim(0, 1.2)
                ax.set_title(f'{bank}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Priority Level', fontsize=12)
                
                # Add recommendation text
                for i, (bar, rec) in enumerate(zip(bars, bank_recs)):
                    # Shorten text for display
                    short_rec = rec[:60] + '...' if len(rec) > 60 else rec
                    ax.text(1.05, bar.get_y() + bar.get_height()/2., 
                           short_rec, ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved recommendations matrix to {save_path}")
        return save_path
    
    def generate_all_visualizations(self):
        """Generate all required visualizations"""
        print("Generating all visualizations for Task 4...")
        
        # Ensure reports/figures directory exists
        import os
        os.makedirs('reports/figures', exist_ok=True)
        
        visualizations = {}
        
        # Generate each visualization
        visualizations['rating_distribution'] = self.plot_rating_distribution()
        visualizations['sentiment_comparison'] = self.plot_sentiment_comparison()
        visualizations['theme_analysis'] = self.plot_theme_analysis()
        visualizations['wordclouds'] = self.generate_wordclouds()
        visualizations['temporal_trends'] = self.plot_temporal_trends()
        visualizations['recommendations_matrix'] = self.plot_recommendations_matrix()
        
        print("✅ All visualizations generated successfully!")
        return visualizations
