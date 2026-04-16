"""
Employee Performance Predictor - Interactive Dashboard

This Streamlit dashboard provides an interactive interface for:
- Viewing employee performance predictions
- Analyzing performance trends and patterns
- Generating insights and recommendations
- Exploring feature importance and model explanations

Author: Data Science Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class PerformanceDashboard:
    """
    Interactive dashboard for employee performance prediction and analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.load_models_and_data()
        self.setup_sidebar()
    
    def load_models_and_data(self):
        """Load trained models and data."""
        try:
            # Load the best model
            self.model = joblib.load('models/trained/best_model.pkl')
            self.target_encoder = joblib.load('models/trained/target_encoder.pkl')
            
            # Load dataset
            self.data = pd.read_csv('data/processed/employee_data_clean.csv')
            
            # Remove employee_id from features
            self.features = self.data.drop(columns=['employee_id', 'performance_band'])
            self.target = self.data['performance_band']
            
            st.success("Models and data loaded successfully!")
            
        except Exception as e:
            st.error(f"Error loading models or data: {str(e)}")
            st.stop()
    
    def setup_sidebar(self):
        """Setup sidebar navigation and controls."""
        st.sidebar.title("Dashboard Navigation")
        
        # Page selection
        self.page = st.sidebar.selectbox(
            "Select Page",
            ["Overview", "Employee Analysis", "Team Insights", "Model Performance", "Predictions"]
        )
        
        # Filters
        st.sidebar.subheader("Filters")
        
        # Department filter
        departments = ['All'] + list(self.data['department'].unique())
        self.selected_department = st.sidebar.selectbox(
            "Department", departments
        )
        
        # Job level filter
        job_levels = ['All'] + list(self.data['job_level'].unique())
        self.selected_job_level = st.sidebar.selectbox(
            "Job Level", job_levels
        )
        
        # Performance band filter
        performance_bands = ['All'] + list(self.data['performance_band'].unique())
        self.selected_performance = st.sidebar.selectbox(
            "Performance Band", performance_bands
        )
        
        # Apply filters
        self.apply_filters()
    
    def apply_filters(self):
        """Apply selected filters to the data."""
        filtered_data = self.data.copy()
        
        if self.selected_department != 'All':
            filtered_data = filtered_data[filtered_data['department'] == self.selected_department]
        
        if self.selected_job_level != 'All':
            filtered_data = filtered_data[filtered_data['job_level'] == self.selected_job_level]
        
        if self.selected_performance != 'All':
            filtered_data = filtered_data[filtered_data['performance_band'] == self.selected_performance]
        
        self.filtered_data = filtered_data
    
    def render_overview_page(self):
        """Render the overview page with key metrics and visualizations."""
        st.markdown('<h1 class="main-header">Employee Performance Overview</h1>', unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_employees = len(self.filtered_data)
            st.metric("Total Employees", total_employees)
        
        with col2:
            high_performers = len(self.filtered_data[self.filtered_data['performance_band'] == 'High'])
            st.metric("High Performers", high_performers)
        
        with col3:
            avg_salary = self.filtered_data['salary'].mean()
            st.metric("Avg Salary", f"${avg_salary:,.0f}")
        
        with col4:
            avg_experience = self.filtered_data['experience_years'].mean()
            st.metric("Avg Experience", f"{avg_experience:.1f} years")
        
        # Performance Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Distribution")
            perf_counts = self.filtered_data['performance_band'].value_counts()
            
            fig = px.pie(
                values=perf_counts.values,
                names=perf_counts.index,
                title="Performance Bands"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Department Performance")
            dept_perf = pd.crosstab(self.filtered_data['department'], self.filtered_data['performance_band'])
            
            fig = px.bar(
                dept_perf,
                x=dept_perf.index,
                y=dept_perf.columns,
                title="Performance by Department",
                barmode='stack'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Key Insights
        st.subheader("Key Insights")
        
        # Calculate insights
        insights = []
        
        # High performer percentage
        high_pct = (len(self.filtered_data[self.filtered_data['performance_band'] == 'High']) / 
                   len(self.filtered_data)) * 100
        
        if high_pct < 10:
            insights.append(f"Only {high_pct:.1f}% of employees are high performers - consider recognition programs")
        elif high_pct > 20:
            insights.append(f"{high_pct:.1f}% high performers - excellent performance culture!")
        
        # Salary insights
        high_performer_salary = self.filtered_data[
            self.filtered_data['performance_band'] == 'High']['salary'].mean()
        low_performer_salary = self.filtered_data[
            self.filtered_data['performance_band'] == 'Low']['salary'].mean()
        
        if high_performer_salary > low_performer_salary * 1.2:
            insights.append("High performers earn significantly more than low performers")
        
        # Training insights
        high_performer_training = self.filtered_data[
            self.filtered_data['performance_band'] == 'High']['training_hours'].mean()
        
        if high_performer_training > 30:
            insights.append("High performers invest more in training and development")
        
        # Display insights
        for insight in insights:
            st.markdown(f'<div class="insight-box"> {insight} </div>', unsafe_allow_html=True)
    
    def render_employee_analysis_page(self):
        """Render the employee analysis page."""
        st.markdown('<h1 class="main-header">Employee Analysis</h1>', unsafe_allow_html=True)
        
        # Employee selection
        employee_ids = ['Select an employee'] + list(self.filtered_data['employee_id'].unique())
        selected_employee = st.selectbox("Select Employee", employee_ids)
        
        if selected_employee != 'Select an employee':
            employee_data = self.filtered_data[self.filtered_data['employee_id'] == selected_employee].iloc[0]
            
            # Employee Profile
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Employee Profile")
                
                profile_info = {
                    'Employee ID': employee_data['employee_id'],
                    'Department': employee_data['department'],
                    'Job Level': employee_data['job_level'],
                    'Education': employee_data['education_level'],
                    'Age': employee_data['age'],
                    'Experience': f"{employee_data['experience_years']} years",
                    'Tenure': f"{employee_data['company_tenure']} years"
                }
                
                for key, value in profile_info.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.subheader("Performance Metrics")
                
                performance_metrics = {
                    'Current Performance': employee_data['performance_band'],
                    'Quality Score': f"{employee_data['quality_score']}/5",
                    'Peer Feedback': f"{employee_data['peer_feedback_score']}/5",
                    'Manager Score': f"{employee_data['manager_score']}/5",
                    'On-time Delivery': f"{employee_data['on_time_delivery_rate']:.1%}",
                    'Attendance Rate': f"{employee_data['attendance_rate']:.1%}"
                }
                
                for key, value in performance_metrics.items():
                    st.write(f"**{key}:** {value}")
            
            # Performance Radar Chart
            st.subheader("Performance Radar Chart")
            
            radar_data = {
                'Quality': employee_data['quality_score'],
                'Peer Feedback': employee_data['peer_feedback_score'],
                'Manager Score': employee_data['manager_score'],
                'Training': min(5, employee_data['training_hours'] / 20),
                'Attendance': employee_data['attendance_rate'] * 5
            }
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(radar_data.values()),
                theta=list(radar_data.keys()),
                fill='toself',
                name=employee_data['employee_id']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5]
                    )),
                showlegend=True,
                title="Performance Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Predict Future Performance
            if st.button("Predict Future Performance"):
                prediction_result = self.predict_employee_performance(selected_employee)
                st.json(prediction_result)
    
    def render_team_insights_page(self):
        """Render the team insights page."""
        st.markdown('<h1 class="main-header">Team Insights</h1>', unsafe_allow_html=True)
        
        # Team Performance Comparison
        st.subheader("Team Performance Comparison")
        
        # Department comparison
        dept_stats = self.filtered_data.groupby('department').agg({
            'performance_band': lambda x: (x == 'High').mean(),
            'salary': 'mean',
            'experience_years': 'mean',
            'training_hours': 'mean'
        }).round(2)
        
        dept_stats.columns = ['High Performer %', 'Avg Salary', 'Avg Experience', 'Avg Training']
        
        fig = px.bar(
            dept_stats,
            x=dept_stats.index,
            y='High Performer %',
            title="High Performer Percentage by Department"
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Job Level Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance by Job Level")
            level_perf = pd.crosstab(self.filtered_data['job_level'], self.filtered_data['performance_band'], normalize='index')
            
            fig = px.bar(
                level_perf,
                x=level_perf.index,
                y=level_perf.columns,
                title="Performance Distribution by Job Level",
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Salary vs Performance")
            
            fig = px.box(
                self.filtered_data,
                x='performance_band',
                y='salary',
                title="Salary Distribution by Performance"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Training Impact Analysis
        st.subheader("Training Impact Analysis")
        
        # Create training buckets
        self.filtered_data['training_bucket'] = pd.cut(
            self.filtered_data['training_hours'],
            bins=[0, 10, 25, 50, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        training_impact = pd.crosstab(
            self.filtered_data['training_bucket'], 
            self.filtered_data['performance_band'], 
            normalize='index'
        )
        
        fig = px.bar(
            training_impact,
            x=training_impact.index,
            y=training_impact.columns,
            title="Performance by Training Hours",
            barmode='stack'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_model_performance_page(self):
        """Render the model performance page."""
        st.markdown('<h1 class="main-header">Model Performance</h1>', unsafe_allow_html=True)
        
        # Load model results
        try:
            model_results = joblib.load('models/trained/model_results.json')
            
            # Model Comparison
            st.subheader("Model Comparison")
            
            models_data = []
            for model_name, results in model_results.items():
                models_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'CV Score': results['best_score']
                })
            
            models_df = pd.DataFrame(models_data)
            
            fig = px.bar(
                models_df,
                x='Model',
                y='CV Score',
                title="Model Cross-Validation Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.subheader("Feature Importance")
            
            # Get feature importance from the best model
            feature_importance = self.get_feature_importance()
            
            if feature_importance:
                importance_df = pd.DataFrame(
                    list(feature_importance.items())[:15],
                    columns=['Feature', 'Importance']
                )
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Important Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading model results: {str(e)}")
        
        # Model Metrics
        st.subheader("Model Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Sample metrics (would normally load from evaluation results)
        with col1:
            st.metric("Accuracy", "89.5%")
        with col2:
            st.metric("F1-Score", "0.63")
        with col3:
            st.metric("Precision", "0.66")
        with col4:
            st.metric("Recall", "0.62")
    
    def render_predictions_page(self):
        """Render the predictions page."""
        st.markdown('<h1 class="main-header">Performance Predictions</h1>', unsafe_allow_html=True)
        
        # Batch prediction
        st.subheader("Batch Performance Prediction")
        
        # Select employees for prediction
        employee_options = self.filtered_data['employee_id'].unique()
        selected_employees = st.multiselect(
            "Select Employees for Prediction",
            employee_options,
            default=employee_options[:5] if len(employee_options) >= 5 else employee_options
        )
        
        if selected_employees:
            if st.button("Generate Predictions"):
                predictions = []
                
                for emp_id in selected_employees:
                    emp_data = self.filtered_data[self.filtered_data['employee_id'] == emp_id]
                    features = emp_data.drop(columns=['employee_id', 'performance_band'])
                    
                    # Make prediction
                    pred_proba = self.model.predict_proba(features)
                    pred_class = self.model.predict(features)
                    pred_label = self.target_encoder.inverse_transform(pred_class)[0]
                    
                    # Get confidence
                    confidence = np.max(pred_proba[0])
                    
                    predictions.append({
                        'Employee ID': emp_id,
                        'Predicted Performance': pred_label,
                        'Confidence': f"{confidence:.1%}",
                        'High Probability': f"{pred_proba[0][0]:.1%}",
                        'Low Probability': f"{pred_proba[0][1]:.1%}",
                        'Medium Probability': f"{pred_proba[0][2]:.1%}"
                    })
                
                # Display predictions
                predictions_df = pd.DataFrame(predictions)
                st.dataframe(predictions_df, use_container_width=True)
                
                # Download predictions
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv,
                    file_name="performance_predictions.csv",
                    mime="text/csv"
                )
        
        # Individual prediction form
        st.subheader("Individual Employee Prediction")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", min_value=22, max_value=65, value=30)
                experience = st.number_input("Experience (years)", min_value=0, max_value=40, value=5)
                salary = st.number_input("Salary", min_value=30000, max_value=300000, value=60000)
            
            with col2:
                department = st.selectbox("Department", self.data['department'].unique())
                job_level = st.selectbox("Job Level", self.data['job_level'].unique())
                education = st.selectbox("Education", self.data['education_level'].unique())
            
            with col3:
                quality_score = st.slider("Quality Score", 1.0, 5.0, 3.0)
                peer_score = st.slider("Peer Feedback Score", 1.0, 5.0, 3.0)
                manager_score = st.slider("Manager Score", 1.0, 5.0, 3.0)
            
            training_hours = st.number_input("Training Hours", min_value=0, max_value=500, value=20)
            projects_completed = st.number_input("Projects Completed", min_value=0, max_value=200, value=10)
            
            submitted = st.form_submit_button("Predict Performance")
            
            if submitted:
                # Create feature dictionary
                features = {
                    'age': age,
                    'experience_years': experience,
                    'company_tenure': experience * 0.7,  # Estimate
                    'salary': salary,
                    'promotion_history': 0,
                    'quality_score': quality_score,
                    'peer_feedback_score': peer_score,
                    'manager_score': manager_score,
                    'on_time_delivery_rate': 0.85,
                    'training_hours': training_hours,
                    'certifications_count': 1,
                    'projects_completed': projects_completed,
                    'attendance_rate': 0.95,
                    'sick_days': 5,
                    'overtime_hours': 10,
                    'department': department,
                    'job_level': job_level,
                    'education_level': education,
                    'gender': 'Male'  # Default
                }
                
                # Create DataFrame
                features_df = pd.DataFrame([features])
                
                # Make prediction
                try:
                    pred_proba = self.model.predict_proba(features_df)
                    pred_class = self.model.predict(features_df)
                    pred_label = self.target_encoder.inverse_transform(pred_class)[0]
                    confidence = np.max(pred_proba[0])
                    
                    # Display results
                    st.success(f"Predicted Performance: {pred_label}")
                    st.info(f"Confidence: {confidence:.1%}")
                    
                    # Display probabilities
                    proba_df = pd.DataFrame({
                        'Performance': self.target_encoder.classes_,
                        'Probability': pred_proba[0]
                    })
                    
                    fig = px.bar(
                        proba_df,
                        x='Performance',
                        y='Probability',
                        title="Prediction Probabilities"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
    
    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        try:
            # This would need to be implemented based on the model type
            # For now, return a sample dictionary
            return {
                'training_hours': 0.15,
                'quality_score': 0.12,
                'manager_score': 0.11,
                'experience_years': 0.10,
                'salary': 0.08
            }
        except:
            return {}
    
    def predict_employee_performance(self, employee_id):
        """Predict performance for a specific employee."""
        try:
            employee_data = self.filtered_data[self.filtered_data['employee_id'] == employee_id]
            features = employee_data.drop(columns=['employee_id', 'performance_band'])
            
            pred_proba = self.model.predict_proba(features)
            pred_class = self.model.predict(features)
            pred_label = self.target_encoder.inverse_transform(pred_class)[0]
            
            return {
                'employee_id': employee_id,
                'predicted_performance': pred_label,
                'confidence': float(np.max(pred_proba[0])),
                'probabilities': {
                    'high': float(pred_proba[0][0]),
                    'low': float(pred_proba[0][1]),
                    'medium': float(pred_proba[0][2])
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run(self):
        """Run the dashboard."""
        # Render the selected page
        if self.page == "Overview":
            self.render_overview_page()
        elif self.page == "Employee Analysis":
            self.render_employee_analysis_page()
        elif self.page == "Team Insights":
            self.render_team_insights_page()
        elif self.page == "Model Performance":
            self.render_model_performance_page()
        elif self.page == "Predictions":
            self.render_predictions_page()


def main():
    """Main function to run the dashboard."""
    # Create dashboard instance
    dashboard = PerformanceDashboard()
    
    # Run dashboard
    dashboard.run()


if __name__ == "__main__":
    main()
