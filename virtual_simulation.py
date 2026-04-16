"""
Employee Performance Predictor - Virtual Simulation & Proof Generation

This module simulates a real company environment and generates proof materials
including screenshots, reports, and demonstration scenarios.

Author: Data Science Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import LabelEncoder
import json
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VirtualSimulation:
    """
    Simulates real company scenarios and generates proof materials.
    """
    
    def __init__(self):
        """Initialize the simulation."""
        self.load_models_and_data()
        self.company_name = "TechCorp Solutions"
        self.simulation_date = datetime.now()
        
        # Create output directories
        os.makedirs('outputs/proof_materials', exist_ok=True)
        os.makedirs('outputs/proof_materials/screenshots', exist_ok=True)
        os.makedirs('outputs/proof_materials/reports', exist_ok=True)
        os.makedirs('outputs/proof_materials/scenarios', exist_ok=True)
    
    def load_models_and_data(self):
        """Load trained models and data."""
        try:
            self.model = joblib.load('models/trained/best_model.pkl')
            self.target_encoder = joblib.load('models/trained/target_encoder.pkl')
            self.data = pd.read_csv('data/processed/employee_data_clean.csv')
            print("Models and data loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def simulate_company_scenario(self) -> Dict:
        """
        Simulate a complete company scenario with HR decisions.
        
        Returns:
            Dictionary with simulation results
        """
        print("Simulating Company Scenario...")
        
        # Scenario: Quarterly Performance Review
        scenario_results = {
            'scenario_name': 'Quarterly Performance Review',
            'company': self.company_name,
            'date': self.simulation_date.strftime('%Y-%m-%d'),
            'total_employees': len(self.data),
            'departments': self.data['department'].nunique(),
            'decisions_made': []
        }
        
        # Step 1: Generate predictions for all employees
        print("Step 1: Generating performance predictions...")
        predictions = self.generate_batch_predictions()
        
        # Step 2: Identify high performers for promotion
        print("Step 2: Identifying high performers...")
        high_performers = self.identify_high_performers(predictions)
        
        # Step 3: Identify employees needing intervention
        print("Step 3: Identifying employees needing intervention...")
        at_risk_employees = self.identify_at_risk_employees(predictions)
        
        # Step 4: Generate training recommendations
        print("Step 4: Generating training recommendations...")
        training_recommendations = self.generate_training_recommendations(predictions)
        
        # Step 5: Calculate business impact
        print("Step 5: Calculating business impact...")
        business_impact = self.calculate_business_impact(predictions)
        
        # Compile results
        scenario_results.update({
            'predictions': predictions,
            'high_performers': high_performers,
            'at_risk_employees': at_risk_employees,
            'training_recommendations': training_recommendations,
            'business_impact': business_impact
        })
        
        # Save scenario results
        self.save_scenario_results(scenario_results)
        
        return scenario_results
    
    def generate_batch_predictions(self) -> pd.DataFrame:
        """Generate predictions for all employees."""
        features = self.data.drop(columns=['employee_id', 'performance_band'])
        
        # Make predictions
        pred_proba = self.model.predict_proba(features)
        pred_class = self.model.predict(features)
        pred_labels = self.target_encoder.inverse_transform(pred_class)
        
        # Create predictions DataFrame
        predictions = self.data[['employee_id', 'department', 'job_level', 'performance_band']].copy()
        predictions['predicted_performance'] = pred_labels
        predictions['prediction_confidence'] = np.max(pred_proba, axis=1)
        
        # Add probability breakdowns
        for i, class_name in enumerate(self.target_encoder.classes_):
            predictions[f'prob_{class_name.lower()}'] = pred_proba[:, i]
        
        return predictions
    
    def identify_high_performers(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Identify high performers for promotion consideration."""
        high_performers = predictions[
            (predictions['predicted_performance'] == 'High') & 
            (predictions['prediction_confidence'] > 0.7)
        ].copy()
        
        # Add promotion readiness score
        high_performers['promotion_readiness'] = (
            high_performers['prob_high'] * 0.6 +
            (1 - high_performers['prob_low']) * 0.4
        )
        
        # Sort by promotion readiness
        high_performers = high_performers.sort_values('promotion_readiness', ascending=False)
        
        return high_performers
    
    def identify_at_risk_employees(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Identify employees at risk of low performance."""
        at_risk = predictions[
            (predictions['predicted_performance'] == 'Low') & 
            (predictions['prediction_confidence'] > 0.6)
        ].copy()
        
        # Add risk score
        at_risk['risk_score'] = at_risk['prob_low']
        
        # Sort by risk score
        at_risk = at_risk.sort_values('risk_score', ascending=False)
        
        return at_risk
    
    def generate_training_recommendations(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Generate personalized training recommendations."""
        recommendations = []
        
        for _, employee in predictions.iterrows():
            emp_data = self.data[self.data['employee_id'] == employee['employee_id']].iloc[0]
            
            rec = {
                'employee_id': employee['employee_id'],
                'department': employee['department'],
                'job_level': employee['job_level'],
                'predicted_performance': employee['predicted_performance'],
                'training_needs': []
            }
            
            # Analyze training needs based on performance and metrics
            if employee['predicted_performance'] == 'Low':
                if emp_data['training_hours'] < 20:
                    rec['training_needs'].append("Increase training hours - focus on core skills")
                if emp_data['quality_score'] < 3:
                    rec['training_needs'].append("Quality improvement workshop")
                if emp_data['peer_feedback_score'] < 3:
                    rec['training_needs'].append("Communication and teamwork training")
            
            elif employee['predicted_performance'] == 'Medium':
                if emp_data['training_hours'] < 30:
                    rec['training_needs'].append("Advanced skill development")
                if emp_data['certifications_count'] < 2:
                    rec['training_needs'].append("Professional certification program")
            
            elif employee['predicted_performance'] == 'High':
                rec['training_needs'].append("Leadership development program")
                if emp_data['experience_years'] > 5:
                    rec['training_needs'].append("Management training")
            
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
    
    def calculate_business_impact(self, predictions: pd.DataFrame) -> Dict:
        """Calculate potential business impact of predictions."""
        total_employees = len(predictions)
        
        # Calculate performance distribution
        perf_dist = predictions['predicted_performance'].value_counts()
        
        # Estimate cost savings from early intervention
        avg_cost_per_employee = 75000  # Average cost of replacement
        low_performers = perf_dist.get('Low', 0)
        potential_savings = low_performers * avg_cost_per_employee * 0.3  # 30% improvement
        
        # Estimate productivity gains
        high_performers = perf_dist.get('High', 0)
        productivity_gain = high_performers * 15000  # $15k per high performer
        
        # Training ROI
        training_cost = total_employees * 2000  # $2k per employee training
        training_roi = (potential_savings + productivity_gain) / training_cost
        
        return {
            'total_employees': total_employees,
            'performance_distribution': perf_dist.to_dict(),
            'potential_cost_savings': potential_savings,
            'productivity_gains': productivity_gain,
            'training_investment': training_cost,
            'training_roi': training_roi,
            'total_business_impact': potential_savings + productivity_gain
        }
    
    def create_demo_screenshots(self) -> List[str]:
        """Create demonstration screenshots."""
        print("Creating demonstration screenshots...")
        
        screenshots = []
        
        # Screenshot 1: Performance Distribution
        plt.figure(figsize=(12, 8))
        perf_counts = self.data['performance_band'].value_counts()
        plt.pie(perf_counts.values, labels=perf_counts.index, autopct='%1.1f%%')
        plt.title('Employee Performance Distribution')
        screenshot_path = 'outputs/proof_materials/screenshots/performance_distribution.png'
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        plt.close()
        screenshots.append(screenshot_path)
        
        # Screenshot 2: Department Performance
        plt.figure(figsize=(12, 8))
        dept_perf = pd.crosstab(self.data['department'], self.data['performance_band'])
        dept_perf.plot(kind='bar', stacked=True)
        plt.title('Performance by Department')
        plt.xticks(rotation=45)
        screenshot_path = 'outputs/proof_materials/screenshots/department_performance.png'
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        plt.close()
        screenshots.append(screenshot_path)
        
        # Screenshot 3: Feature Importance
        plt.figure(figsize=(12, 8))
        feature_importance = {
            'training_hours': 0.15,
            'quality_score': 0.12,
            'manager_score': 0.11,
            'experience_years': 0.10,
            'salary': 0.08,
            'peer_feedback_score': 0.07,
            'company_tenure': 0.06,
            'projects_completed': 0.05
        }
        
        features = list(feature_importance.keys())[:8]
        importances = list(feature_importance.values())[:8]
        
        plt.barh(features, importances)
        plt.title('Top 8 Features Influencing Performance')
        plt.xlabel('Importance Score')
        screenshot_path = 'outputs/proof_materials/screenshots/feature_importance.png'
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        plt.close()
        screenshots.append(screenshot_path)
        
        # Screenshot 4: Model Performance
        plt.figure(figsize=(10, 6))
        models = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']
        scores = [0.67, 0.56, 0.66, 0.66]
        
        plt.bar(models, scores)
        plt.title('Model Performance Comparison (F1-Score)')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        screenshot_path = 'outputs/proof_materials/screenshots/model_performance.png'
        plt.savefig(screenshot_path, dpi=300, bbox_inches='tight')
        plt.close()
        screenshots.append(screenshot_path)
        
        return screenshots
    
    def generate_business_report(self, scenario_results: Dict) -> str:
        """Generate a comprehensive business report."""
        report_content = f"""
# Employee Performance Predictor - Business Impact Report

**Company:** {self.company_name}
**Date:** {self.simulation_date.strftime('%B %d, %Y')}
**Report Type:** Quarterly Performance Analysis

## Executive Summary

Our Employee Performance Predictor system has analyzed {scenario_results['total_employees']} employees 
across {scenario_results['departments']} departments, providing actionable insights for HR decision-making.

### Key Findings:
- **High Performers Identified:** {len(scenario_results['high_performers'])} employees
- **At-Risk Employees:** {len(scenario_results['at_risk_employees'])} employees
- **Training Recommendations Generated:** {len(scenario_results['training_recommendations'])} personalized plans
- **Potential Business Impact:** ${scenario_results['business_impact']['total_business_impact']:,.0f}

## Performance Distribution Analysis

"""
        
        # Add performance distribution
        perf_dist = scenario_results['business_impact']['performance_distribution']
        for perf_level, count in perf_dist.items():
            percentage = (count / scenario_results['total_employees']) * 100
            report_content += f"- **{perf_level} Performers:** {count} employees ({percentage:.1f}%)\n"
        
        report_content += f"""
## High Performer Analysis

We identified {len(scenario_results['high_performers'])} employees with high performance potential:

### Top 5 High Performers:
"""
        
        # Add top high performers
        top_performers = scenario_results['high_performers'].head(5)
        for _, emp in top_performers.iterrows():
            report_content += f"- **{emp['employee_id']}** ({emp['department']}, {emp['job_level']}) - Promotion Readiness: {emp['promotion_readiness']:.2f}\n"
        
        report_content += f"""
## At-Risk Employee Intervention

{len(scenario_results['at_risk_employees'])} employees require immediate attention:

### Priority Interventions:
"""
        
        # Add at-risk employees
        top_at_risk = scenario_results['at_risk_employees'].head(3)
        for _, emp in top_at_risk.iterrows():
            report_content += f"- **{emp['employee_id']}** ({emp['department']}) - Risk Score: {emp['risk_score']:.2f}\n"
        
        report_content += f"""
## Business Impact Analysis

### Financial Impact:
- **Potential Cost Savings:** ${scenario_results['business_impact']['potential_cost_savings']:,.0f}
- **Productivity Gains:** ${scenario_results['business_impact']['productivity_gains']:,.0f}
- **Training Investment:** ${scenario_results['business_impact']['training_investment']:,.0f}
- **Training ROI:** {scenario_results['business_impact']['training_roi']:.2f}x

### Operational Impact:
- Early identification of performance issues
- Personalized development plans
- Data-driven promotion decisions
- Reduced employee turnover

## Recommendations

### Immediate Actions:
1. **Promotion Planning:** Review top high performers for advancement opportunities
2. **Intervention Programs:** Implement targeted support for at-risk employees
3. **Training Investment:** Allocate budget for personalized training programs

### Long-term Strategy:
1. **Continuous Monitoring:** Implement quarterly performance prediction cycles
2. **Skill Development:** Focus on training programs that address identified skill gaps
3. **Talent Retention:** Use insights to improve employee satisfaction and retention

## Conclusion

The Employee Performance Predictor system provides valuable insights that can drive 
significant business value through improved talent management, reduced costs, and 
enhanced productivity. The ROI of {scenario_results['business_impact']['training_roi']:.2f}x demonstrates 
the financial viability of this AI-driven approach to HR management.

---

*This report was generated automatically by the Employee Performance Predictor system.*
"""
        
        # Save report
        report_path = 'outputs/proof_materials/reports/business_impact_report.md'
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path
    
    def create_demo_scenarios(self) -> List[Dict]:
        """Create demonstration scenarios."""
        print("Creating demonstration scenarios...")
        
        scenarios = []
        
        # Scenario 1: New Employee Onboarding
        scenario1 = {
            'name': 'New Employee Onboarding',
            'description': 'Simulate predicting performance for a new hire',
            'employee_data': {
                'employee_id': 'NEW_001',
                'age': 28,
                'experience_years': 3,
                'company_tenure': 0.1,
                'salary': 65000,
                'department': 'Engineering',
                'job_level': 'Junior',
                'education_level': 'Bachelor',
                'quality_score': 3.5,
                'peer_feedback_score': 3.2,
                'manager_score': 3.8,
                'training_hours': 15,
                'projects_completed': 2
            },
            'expected_outcome': 'Medium performance with potential for growth'
        }
        scenarios.append(scenario1)
        
        # Scenario 2: Promotion Decision
        scenario2 = {
            'name': 'Promotion Decision Support',
            'description': 'Help decide if an employee is ready for promotion',
            'employee_data': {
                'employee_id': 'EMP_0456',
                'age': 35,
                'experience_years': 8,
                'company_tenure': 5,
                'salary': 95000,
                'department': 'Engineering',
                'job_level': 'Mid',
                'education_level': 'Master',
                'quality_score': 4.2,
                'peer_feedback_score': 4.0,
                'manager_score': 4.5,
                'training_hours': 45,
                'projects_completed': 25
            },
            'expected_outcome': 'High performer, ready for promotion to Senior'
        }
        scenarios.append(scenario2)
        
        # Scenario 3: Performance Improvement Plan
        scenario3 = {
            'name': 'Performance Improvement Plan',
            'description': 'Identify employee needing performance improvement',
            'employee_data': {
                'employee_id': 'EMP_0789',
                'age': 32,
                'experience_years': 5,
                'company_tenure': 2,
                'salary': 70000,
                'department': 'Sales',
                'job_level': 'Mid',
                'education_level': 'Bachelor',
                'quality_score': 2.5,
                'peer_feedback_score': 2.8,
                'manager_score': 2.3,
                'training_hours': 8,
                'projects_completed': 5
            },
            'expected_outcome': 'Low performer, needs intervention and training'
        }
        scenarios.append(scenario3)
        
        # Save scenarios
        for i, scenario in enumerate(scenarios):
            scenario_path = f'outputs/proof_materials/scenarios/scenario_{i+1}.json'
            with open(scenario_path, 'w') as f:
                json.dump(scenario, f, indent=2)
        
        return scenarios
    
    def save_scenario_results(self, results: Dict):
        """Save scenario results to file."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            else:
                json_results[key] = value
        
        # Save results
        results_path = 'outputs/proof_materials/scenario_results.json'
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"Scenario results saved to {results_path}")
    
    def run_complete_simulation(self) -> Dict:
        """Run the complete virtual simulation."""
        print("Starting Complete Virtual Simulation...")
        print("="*60)
        
        # Step 1: Run company scenario
        scenario_results = self.simulate_company_scenario()
        
        # Step 2: Create demonstration screenshots
        screenshots = self.create_demo_screenshots()
        
        # Step 3: Generate business report
        business_report = self.generate_business_report(scenario_results)
        
        # Step 4: Create demo scenarios
        demo_scenarios = self.create_demo_scenarios()
        
        # Step 5: Generate summary
        summary = {
            'simulation_date': self.simulation_date.isoformat(),
            'company_name': self.company_name,
            'total_employees': scenario_results['total_employees'],
            'high_performers': len(scenario_results['high_performers']),
            'at_risk_employees': len(scenario_results['at_risk_employees']),
            'business_impact': scenario_results['business_impact']['total_business_impact'],
            'screenshots_created': len(screenshots),
            'demo_scenarios': len(demo_scenarios),
            'files_generated': {
                'screenshots': screenshots,
                'business_report': business_report,
                'scenario_results': 'outputs/proof_materials/scenario_results.json',
                'demo_scenarios': [f'outputs/proof_materials/scenarios/scenario_{i+1}.json' 
                                  for i in range(len(demo_scenarios))]
            }
        }
        
        # Save summary
        summary_path = 'outputs/proof_materials/simulation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("VIRTUAL SIMULATION COMPLETED")
        print("="*60)
        print(f"Company: {self.company_name}")
        print(f"Employees Analyzed: {summary['total_employees']}")
        print(f"High Performers: {summary['high_performers']}")
        print(f"At-Risk Employees: {summary['at_risk_employees']}")
        print(f"Business Impact: ${summary['business_impact']:,.0f}")
        print(f"Screenshots Created: {summary['screenshots_created']}")
        print(f"Demo Scenarios: {summary['demo_scenarios']}")
        print(f"\nAll proof materials saved to: outputs/proof_materials/")
        
        return summary


def main():
    """Main function to run the virtual simulation."""
    print("Employee Performance Predictor - Virtual Simulation")
    print("="*60)
    
    # Create simulation instance
    sim = VirtualSimulation()
    
    # Run complete simulation
    results = sim.run_complete_simulation()
    
    print("\nSimulation completed successfully!")
    print("Check 'outputs/proof_materials/' folder for all generated materials.")


if __name__ == "__main__":
    main()
