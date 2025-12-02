import sys
import os
import json
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sentiment_analyzer import SentimentAnalyzer
from scripts.thematic_analyzer import ThematicAnalyzer
import logging

def main():
    """Execute Task 2: Sentiment and Thematic Analysis"""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("TASK 2: SENTIMENT AND THEMATIC ANALYSIS")
    print("=" * 60)
    
    # Step 1: Load processed data
    print("\n1. Loading processed data...")
    try:
        df = pd.read_csv('data/processed/reviews_processed.csv')
        print(f"✓ Loaded {len(df)} reviews")
    except FileNotFoundError:
        print("✗ Processed data not found. Run Task 1 first.")
        return
    
    # Step 2: Perform sentiment analysis
    print("\n2. Performing sentiment analysis...")
    sentiment_analyzer = SentimentAnalyzer()
    df_with_sentiment = sentiment_analyzer.analyze_dataframe(df)
    
    # Save sentiment analysis results
    sentiment_file = 'data/processed/reviews_with_sentiment.csv'
    df_with_sentiment.to_csv(sentiment_file, index=False)
    print(f"✓ Sentiment analysis complete. Saved to: {sentiment_file}")
    
    # Generate sentiment summary
    sentiment_summary = sentiment_analyzer.generate_sentiment_summary(df_with_sentiment)
    
    # Step 3: Perform thematic analysis
    print("\n3. Performing thematic analysis...")
    thematic_analyzer = ThematicAnalyzer()
    thematic_results = thematic_analyzer.analyze_themes_by_bank(df_with_sentiment)
    
    # Save thematic analysis results
    thematic_file = 'data/processed/thematic_analysis.json'
    thematic_analyzer.save_thematic_analysis(thematic_results, thematic_file)
    print(f"✓ Thematic analysis complete. Saved to: {thematic_file}")
    
    # Step 4: Generate word clouds
    print("\n4. Generating visualizations...")
    for bank in df_with_sentiment['bank'].unique():
        bank_texts = df_with_sentiment[df_with_sentiment['bank'] == bank]['cleaned_text'].tolist()
        save_path = f'data/processed/wordcloud_{bank}.png'
        thematic_analyzer.generate_wordcloud(bank_texts, bank, save_path)
        print(f"  ✓ Word cloud for {bank}: {save_path}")
    
    # Step 5: Save combined results
    print("\n5. Saving comprehensive results...")
    
    # Create comprehensive output
    comprehensive_results = {
        'metadata': {
            'total_reviews': len(df_with_sentiment),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'banks_analyzed': df_with_sentiment['bank'].unique().tolist()
        },
        'sentiment_summary': sentiment_summary,
        'thematic_analysis': thematic_results
    }
    
    # Save comprehensive results
    comprehensive_file = 'data/processed/comprehensive_analysis.json'
    with open(comprehensive_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Comprehensive analysis saved to: {comprehensive_file}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TASK 2 COMPLETE - SUMMARY")
    print("=" * 60)
    
    print(f"\nSentiment Analysis Coverage: {len(df_with_sentiment)} reviews (100%)")
    print(f"Themes Identified per Bank:")
    
    for bank in df_with_sentiment['bank'].unique():
        bank_themes = thematic_results.get(bank, {}).get('theme_analysis', {})
        print(f"  {bank}: {len(bank_themes)} themes identified")
        
        # Print top 3 themes
        most_common = thematic_results[bank]['most_common_themes'][:3]
        for theme, count in most_common:
            percentage = (count / thematic_results[bank]['total_reviews']) * 100
            print(f"    - {theme}: {percentage:.1f}% of reviews")

if __name__ == "__main__":
    main()