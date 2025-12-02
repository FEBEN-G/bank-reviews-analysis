#!/usr/bin/env python3
"""
Run Task 4: Insights and Recommendations
"""
import subprocess
import sys
import os

def run_task4():
    print("🚀 Starting Task 4: Insights and Recommendations")
    print("=" * 60)
    
    # Check if required data exists
    if not os.path.exists('data/processed/combined_reviews.csv'):
        print("❌ Error: Processed data not found.")
        print("   Please run Task 2 first to generate combined_reviews.csv")
        sys.exit(1)
    
    # Step 1: Run insights analysis
    print("\n📊 Step 1: Running insights analysis...")
    result = subprocess.run([sys.executable, 'scripts/task4_main.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error in insights analysis: {result.stderr}")
        sys.exit(1)
    
    # Step 2: Generate PDF report
    print("\n📄 Step 2: Generating PDF report...")
    result = subprocess.run([sys.executable, 'scripts/report_generator.py'], 
                          capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error in report generation: {result.stderr}")
        sys.exit(1)
    
    # Step 3: Create summary
    print("\n✅ Task 4 Completed Successfully!")
    print("=" * 60)
    print("\n📁 Generated Files:")
    print("   • reports/task4_insights_report.json")
    print("   • reports/task4_executive_summary.md")
    print("   • reports/final_report.pdf (10 pages)")
    print("   • reports/figures/ (6 visualization files)")
    
    print("\n📊 Key Statistics:")
    # Count reviews
    import pandas as pd
    df = pd.read_csv('data/processed/combined_reviews.csv')
    print(f"   • Total Reviews: {len(df):,}")
    print(f"   • Banks Analyzed: 3 (CBE, BOA, Dashen)")
    print(f"   • Time Period: {df['date'].min()} to {df['date'].max()}")
    
    print("\n🎯 Next: Commit Task 4 to GitHub")
    print("   git checkout -b task-4")
    print("   git add scripts/insights_analyzer.py scripts/visualization_generator.py")
    print("   git add scripts/task4_main.py scripts/report_generator.py")
    print("   git add reports/")
    print("   git commit -m 'Task 4: Insights and recommendations'")
    print("   git push origin task-4")

if __name__ == "__main__":
    run_task4()
