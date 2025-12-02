#!/usr/bin/env python3
"""
Simple Task 4 Runner
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.insights_analyzer_fixed import InsightsAnalyzerFixed

def main():
    print("🚀 Running Task 4 (Simplified)")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists('data/processed/combined_reviews.csv'):
        print("❌ combined_reviews.csv not found")
        print("   Running data preparation...")
        import subprocess
        result = subprocess.run([sys.executable, 'scripts/prepare_task4_data.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"❌ Preparation failed: {result.stderr}")
            return
    
    # Run analysis
    analyzer = InsightsAnalyzerFixed()
    
    if not analyzer.load_data():
        return
    
    print("\n📊 Calculating metrics...")
    metrics = analyzer.calculate_metrics()
    
    print("\n🔍 Analyzing insights...")
    insights = analyzer.analyze_drivers_pain_points()
    
    print("\n💡 Generating recommendations...")
    recommendations = analyzer.generate_recommendations(insights)
    
    print("\n💾 Saving report...")
    report = analyzer.save_report(metrics, insights, recommendations)
    
    print("\n" + "=" * 60)
    print("✅ TASK 4 COMPLETED!")
    print("=" * 60)
    
    # Print summary
    print("\n📋 SUMMARY:")
    for bank in analyzer.banks:
        print(f"\n{bank}:")
        print(f"  • Reviews: {metrics[bank]['total_reviews']}")
        print(f"  • Avg Rating: {metrics[bank]['avg_rating']:.1f}★")
        print(f"  • Positive: {metrics[bank]['positive_pct']}%")
        print(f"  • Drivers: {', '.join(insights[bank]['drivers'][:3])}")
        print(f"  • Pain Points: {', '.join(insights[bank]['pain_points'][:3])}")
        print(f"  • Recommendations: {recommendations[bank][0]}")

if __name__ == "__main__":
    main()
