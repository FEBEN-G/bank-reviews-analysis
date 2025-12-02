"""
Prepare data for Task 4 from Task 2 output
"""
import pandas as pd
import numpy as np
import json

def main():
    print("Preparing data for Task 4...")
    
    # Load the data
    df = pd.read_csv('data/processed/reviews_with_themes.csv')
    print(f"Loaded {len(df)} reviews")
    
    # Create a primary_theme column based on theme columns
    theme_columns = [
        'theme_login_issues',
        'theme_transaction_problems', 
        'theme_app_performance',
        'theme_user_interface',
        'theme_customer_support',
        'theme_security_concerns',
        'theme_feature_requests',
        'theme_account_management'
    ]
    
    # Check which theme columns exist
    existing_theme_cols = [col for col in theme_columns if col in df.columns]
    print(f"Found theme columns: {existing_theme_cols}")
    
    # Function to get primary theme for each row
    def get_primary_theme(row):
        for theme_col in existing_theme_cols:
            if row[theme_col] == 1:
                # Extract theme name from column name
                theme_name = theme_col.replace('theme_', '').replace('_', ' ').title()
                return theme_name
        return 'General Feedback'
    
    # Apply function to create primary_theme column
    df['primary_theme'] = df.apply(get_primary_theme, axis=1)
    
    # Create cleaned text column
    df['review_text_clean'] = df['review'].astype(str).str.lower().str.replace('[^\w\s]', ' ', regex=True)
    
    # Ensure sentiment_label is string
    df['sentiment_label'] = df['sentiment_label'].astype(str).str.upper()
    
    # Rename columns for consistency
    if 'review' in df.columns and 'review_text' not in df.columns:
        df = df.rename(columns={'review': 'review_text'})
    
    # Select and reorder columns for Task 4
    task4_columns = [
        'review_text', 'review_text_clean', 'rating', 'date', 'bank', 'source',
        'sentiment_label', 'sentiment_score', 'primary_theme'
    ]
    
    # Add theme columns if they exist
    for col in existing_theme_cols:
        if col not in task4_columns:
            task4_columns.append(col)
    
    # Create final dataframe
    final_df = df[task4_columns]
    
    # Save the prepared data
    output_path = 'data/processed/combined_reviews.csv'
    final_df.to_csv(output_path, index=False)
    
    print(f"✅ Prepared data saved to {output_path}")
    print(f"   Rows: {len(final_df)}")
    print(f"   Columns: {len(final_df.columns)}")
    print(f"   Primary theme distribution:")
    print(final_df['primary_theme'].value_counts().to_string())
    
    return final_df

if __name__ == "__main__":
    main()
