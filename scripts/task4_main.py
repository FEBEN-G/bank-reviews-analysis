"""
Task 4 Main Script: Insights and Recommendations
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.insights_analyzer import InsightsAnalyzer
from scripts.visualization_generator import VisualizationGenerator
import json
from datetime import datetime

def main():
    print("=" * 60)
    print("TASK 4: INSIGHTS AND RECOMMENDATIONS")
    print("=" * 60)
    
    # Step 1: Load and analyze data
    print("\n📊 Step 1: Loading and analyzing data...")
    analyzer = InsightsAnalyzer('data/processed/combined_reviews.csv')
    
    if not analyzer.load_data():
        print("❌ Failed to load data. Please run Task 2 first.")
        return
    
    # Step 2: Calculate metrics
    print("\n📈 Step 2: Calculating key metrics...")
    metrics = analyzer.calculate_key_metrics()
    print(f"   Analyzed {len(analyzer.df)} reviews across {len(analyzer.banks)} banks")
    
    # Step 3: Identify drivers and pain points
    print("\n🔍 Step 3: Identifying drivers and pain points...")
    insights = analyzer.identify_drivers_pain_points()
    
    # Step 4: Generate recommendations
    print("\n💡 Step 4: Generating recommendations...")
    recommendations = analyzer.generate_recommendations(insights)
    
    # Step 5: Save insights report
    print("\n💾 Step 5: Saving insights report...")
    report = analyzer.save_insights_report(metrics, insights, recommendations)
    
    # Step 6: Generate visualizations
    print("\n🎨 Step 6: Generating visualizations...")
    visualizer = VisualizationGenerator('data/processed/combined_reviews.csv')
    visualizer.load_data()
    visualizations = visualizer.generate_all_visualizations()
    
    # Step 7: Generate final summary
    print("\n📋 Step 7: Generating final summary...")
    generate_final_summary(report, visualizations)
    
    print("\n" + "=" * 60)
    print("✅ TASK 4 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Outputs generated:")
    print("   • reports/task4_insights_report.json")
    print("   • reports/figures/ (6 visualization files)")
    print("   • reports/task4_executive_summary.md")
    print("\n📊 Key insights available in the reports directory")

def generate_final_summary(report, visualizations):
    """Generate executive summary markdown"""
    summary = f"""# Executive Summary: Bank Reviews Analysis

## Project Overview
- **Project**: Customer Experience Analytics for Ethiopian Bank Apps
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Total Reviews Analyzed**: {report['total_reviews_analyzed']:,}
- **Banks Analyzed**: {', '.join(report['banks_analyzed'])}

## Key Findings

### 1. Overall Performance
"""
    
    # Add metrics
    for bank in report['banks_analyzed']:
        metrics = report['key_metrics'][bank]
        summary += f"- **{bank}**: {metrics['avg_rating']:.1f}★ average rating | "
        summary += f"{metrics['positive_pct']:.1f}% positive sentiment\n"
    
    summary += """
### 2. Primary Themes Identified
"""
    
    for bank in report['banks_analyzed']:
        themes = list(report['insights'][bank]['top_themes'])
        summary += f"- **{bank}**: {', '.join(themes[:3])}\n"
    
    summary += """
### 3. Critical Insights

#### Scenario 1: Retaining Users
"""
    for analysis in report['scenario_analysis']['scenario_1_retaining_users']['analysis']:
        summary += f"- {analysis}\n"
    
    summary += """
#### Scenario 2: Enhancing Features
"""
    for analysis in report['scenario_analysis']['scenario_2_enhancing_features']['analysis']:
        summary += f"- {analysis}\n"
    
    summary += """
#### Scenario 3: Managing Complaints
"""
    for analysis in report['scenario_analysis']['scenario_3_managing_complaints']['analysis']:
        summary += f"- {analysis}\n"
    
    summary += """
### 4. Priority Recommendations
"""
    
    for bank in report['banks_analyzed']:
        summary += f"\n#### {bank}\n"
        for i, rec in enumerate(report['recommendations'][bank][:3], 1):
            summary += f"{i}. {rec}\n"
    
    summary += f"""
## Visualizations Generated
{len(visualizations)} key visualizations created:
1. Rating Distribution by Bank
2. Sentiment Comparison
3. Theme Analysis
4. Review Word Clouds
5. Temporal Trends
6. Recommendations Matrix

## Next Steps
1. Present findings to bank stakeholders
2. Implement high-priority recommendations
3. Monitor improvements through ongoing review analysis
4. Expand analysis to include more banks and longer time periods

---

*Report generated automatically as part of Task 4 submission.*
"""
    
    with open('reports/task4_executive_summary.md', 'w') as f:
        f.write(summary)
    
    print("✅ Executive summary saved to reports/task4_executive_summary.md")

if __name__ == "__main__":
    main()
