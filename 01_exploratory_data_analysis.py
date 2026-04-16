"""
Employee Performance Predictor - Exploratory Data Analysis

This module performs comprehensive exploratory data analysis including:
- Data overview and summary statistics
- Distribution analysis
- Correlation analysis
- Feature-target relationships
- Statistical tests
- Visualization generation

Author: Data Science Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ExploratoryDataAnalysis:
    """
    Comprehensive EDA for employee performance prediction.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize EDA with data path.
        
        Args:
            data_path: Path to the cleaned dataset
        """
        self.data = pd.read_csv(data_path)
        self.target_column = 'performance_band'
        
        # Identify column types
        self.numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from numerical columns for analysis
        if self.target_column in self.numerical_columns:
            self.numerical_columns.remove(self.target_column)
        
        # Remove ID columns from analysis
        id_columns = ['employee_id']
        self.analysis_columns = [col for col in self.data.columns if col not in id_columns]
        
        print(f"Data loaded: {self.data.shape}")
        print(f"Numerical columns: {len(self.numerical_columns)}")
        print(f"Categorical columns: {len(self.categorical_columns)}")
    
    def basic_statistics(self) -> pd.DataFrame:
        """
        Generate basic statistics for numerical columns.
        
        Returns:
            DataFrame with basic statistics
        """
        stats_df = self.data[self.numerical_columns].describe().T
        
        # Add additional statistics
        stats_df['skewness'] = self.data[self.numerical_columns].skew()
        stats_df['kurtosis'] = self.data[self.numerical_columns].kurt()
        stats_df['missing_values'] = self.data[self.numerical_columns].isnull().sum()
        stats_df['missing_percentage'] = (stats_df['missing_values'] / len(self.data)) * 100
        
        return stats_df
    
    def target_distribution_analysis(self):
        """
        Analyze target variable distribution.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Target distribution
        target_counts = self.data[self.target_column].value_counts()
        axes[0, 0].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Performance Band Distribution')
        
        # Target count plot
        sns.countplot(data=self.data, x=self.target_column, ax=axes[0, 1])
        axes[0, 1].set_title('Performance Band Counts')
        axes[0, 1].set_xlabel('Performance Band')
        axes[0, 1].set_ylabel('Count')
        
        # Target by department
        dept_performance = pd.crosstab(self.data['department'], self.data[self.target_column])
        dept_performance.plot(kind='bar', stacked=True, ax=axes[1, 0])
        axes[1, 0].set_title('Performance by Department')
        axes[1, 0].set_xlabel('Department')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Target by job level
        level_performance = pd.crosstab(self.data['job_level'], self.data[self.target_column])
        level_performance.plot(kind='bar', stacked=True, ax=axes[1, 1])
        axes[1, 1].set_title('Performance by Job Level')
        axes[1, 1].set_xlabel('Job Level')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return target_counts
    
    def numerical_feature_distributions(self):
        """
        Analyze distributions of numerical features.
        """
        # Select key numerical features for visualization
        key_features = [
            'age', 'experience_years', 'salary', 'quality_score', 
            'peer_feedback_score', 'manager_score', 'training_hours',
            'projects_completed', 'attendance_rate'
        ]
        
        # Filter available features
        available_features = [col for col in key_features if col in self.numerical_columns]
        
        # Create subplots
        n_features = len(available_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                # Histogram with KDE
                sns.histplot(data=self.data, x=feature, kde=True, ax=axes[i])
                axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(feature.replace('_', ' ').title())
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def categorical_feature_analysis(self):
        """
        Analyze categorical features.
        """
        categorical_features = ['department', 'job_level', 'education_level', 'gender']
        available_features = [col for col in categorical_features if col in self.categorical_columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_features):
            if i < len(axes):
                # Count plot
                sns.countplot(data=self.data, x=feature, ax=axes[i])
                axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_analysis(self):
        """
        Perform correlation analysis for numerical features.
        """
        # Calculate correlation matrix
        correlation_matrix = self.data[self.numerical_columns].corr()
        
        # Create heatmap
        plt.figure(figsize=(15, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        return correlation_matrix, high_corr_pairs
    
    def feature_target_relationships(self):
        """
        Analyze relationships between features and target variable.
        """
        # Numerical features vs target
        key_numerical = ['age', 'experience_years', 'salary', 'quality_score', 
                         'peer_feedback_score', 'manager_score', 'training_hours']
        available_numerical = [col for col in key_numerical if col in self.numerical_columns]
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_numerical):
            if i < len(axes):
                # Box plot
                sns.boxplot(data=self.data, x=self.target_column, y=feature, ax=axes[i])
                axes[i].set_title(f'{feature.replace("_", " ").title()} by Performance')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_numerical), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/feature_target_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Categorical features vs target
        categorical_features = ['department', 'job_level', 'education_level']
        available_categorical = [col for col in categorical_features if col in self.categorical_columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(available_categorical):
            if i < len(axes):
                # Stacked bar plot
                pd.crosstab(self.data[feature], self.data[self.target_column]).plot(
                    kind='bar', stacked=True, ax=axes[i]
                )
                axes[i].set_title(f'Performance by {feature.replace("_", " ").title()}')
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_categorical), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/categorical_target_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def mutual_information_analysis(self):
        """
        Calculate mutual information scores for feature importance.
        """
        # Prepare data for mutual information
        X = self.data[self.numerical_columns].fillna(0)
        y = self.data[self.target_column]
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # Plot mutual information scores
        plt.figure(figsize=(12, 8))
        mi_scores.head(15).plot(kind='bar')
        plt.title('Top 15 Features by Mutual Information Score')
        plt.xlabel('Features')
        plt.ylabel('Mutual Information Score')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('outputs/visualizations/mutual_information_scores.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return mi_scores
    
    def statistical_tests(self):
        """
        Perform statistical tests to understand feature significance.
        """
        test_results = {}
        
        # ANOVA test for numerical features
        numerical_features = ['age', 'experience_years', 'salary', 'quality_score', 
                             'peer_feedback_score', 'manager_score', 'training_hours']
        available_numerical = [col for col in numerical_features if col in self.numerical_columns]
        
        for feature in available_numerical:
            # Group data by performance band
            groups = [self.data[self.data[self.target_column] == level][feature] 
                     for level in self.data[self.target_column].unique()]
            
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            test_results[feature] = {
                'test': 'ANOVA',
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Chi-square test for categorical features
        categorical_features = ['department', 'job_level', 'education_level']
        available_categorical = [col for col in categorical_features if col in self.categorical_columns]
        
        for feature in available_categorical:
            # Create contingency table
            contingency_table = pd.crosstab(self.data[feature], self.data[self.target_column])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            test_results[feature] = {
                'test': 'Chi-Square',
                'statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
        
        # Convert to DataFrame for easier viewing
        results_df = pd.DataFrame(test_results).T
        
        return results_df
    
    def generate_insights(self):
        """
        Generate key insights from the EDA.
        """
        insights = []
        
        # Performance distribution insight
        target_counts = self.data[self.target_column].value_counts()
        if target_counts['Medium'] > target_counts['High'] + target_counts['Low']:
            insights.append("Most employees fall into the 'Medium' performance category, suggesting a normal distribution.")
        
        # Department insights
        dept_performance = pd.crosstab(self.data['department'], self.data[self.target_column], normalize='index')
        high_performers_dept = dept_performance['High'].idxmax()
        insights.append(f"The {high_performers_dept} department has the highest proportion of high performers.")
        
        # Job level insights
        level_performance = pd.crosstab(self.data['job_level'], self.data[self.target_column], normalize='index')
        high_performers_level = level_performance['High'].idxmax()
        insights.append(f"{high_performers_level} level employees have the highest proportion of high performers.")
        
        # Salary insights
        salary_by_performance = self.data.groupby(self.target_column)['salary'].mean()
        if salary_by_performance['High'] > salary_by_performance['Low']:
            insights.append("High performers earn significantly more than low performers on average.")
        
        # Training insights
        training_by_performance = self.data.groupby(self.target_column)['training_hours'].mean()
        if training_by_performance['High'] > training_by_performance['Medium']:
            insights.append("High performers tend to complete more training hours than other groups.")
        
        # Experience insights
        exp_by_performance = self.data.groupby(self.target_column)['experience_years'].mean()
        if exp_by_performance['High'] > exp_by_performance['Low']:
            insights.append("High performers generally have more years of experience.")
        
        return insights
    
    def create_summary_dashboard(self):
        """
        Create a comprehensive summary dashboard.
        """
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Performance Distribution', 'Department Performance', 'Job Level Performance',
                'Salary by Performance', 'Training Hours by Performance', 'Experience by Performance',
                'Quality Score Distribution', 'Age Distribution', 'Attendance Rate'
            ],
            specs=[
                [{"type": "pie"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "box"}, {"type": "box"}],
                [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]
            ]
        )
        
        # Performance distribution (pie chart)
        target_counts = self.data[self.target_column].value_counts()
        fig.add_trace(
            go.Pie(labels=target_counts.index, values=target_counts.values, name="Performance"),
            row=1, col=1
        )
        
        # Department performance (bar chart)
        dept_counts = self.data['department'].value_counts()
        fig.add_trace(
            go.Bar(x=dept_counts.index, y=dept_counts.values, name="Department"),
            row=1, col=2
        )
        
        # Job level performance (bar chart)
        level_counts = self.data['job_level'].value_counts()
        fig.add_trace(
            go.Bar(x=level_counts.index, y=level_counts.values, name="Job Level"),
            row=1, col=3
        )
        
        # Salary by performance (box plot)
        for perf in self.data[self.target_column].unique():
            fig.add_trace(
                go.Box(y=self.data[self.data[self.target_column] == perf]['salary'], 
                       name=f"Salary - {perf}"),
                row=2, col=1
            )
        
        # Training hours by performance (box plot)
        for perf in self.data[self.target_column].unique():
            fig.add_trace(
                go.Box(y=self.data[self.data[self.target_column] == perf]['training_hours'], 
                       name=f"Training - {perf}"),
                row=2, col=2
            )
        
        # Experience by performance (box plot)
        for perf in self.data[self.target_column].unique():
            fig.add_trace(
                go.Box(y=self.data[self.data[self.target_column] == perf]['experience_years'], 
                       name=f"Experience - {perf}"),
                row=2, col=3
            )
        
        # Quality score distribution (histogram)
        fig.add_trace(
            go.Histogram(x=self.data['quality_score'], name="Quality Score"),
            row=3, col=1
        )
        
        # Age distribution (histogram)
        fig.add_trace(
            go.Histogram(x=self.data['age'], name="Age"),
            row=3, col=2
        )
        
        # Attendance rate (histogram)
        fig.add_trace(
            go.Histogram(x=self.data['attendance_rate'], name="Attendance Rate"),
            row=3, col=3
        )
        
        fig.update_layout(height=1200, showlegend=False, title_text="Employee Performance EDA Dashboard")
        fig.write_html('outputs/visualizations/eda_dashboard.html')
        fig.show()
    
    def run_complete_eda(self):
        """
        Run the complete EDA pipeline.
        """
        print("Starting Exploratory Data Analysis...")
        
        # Basic statistics
        print("\n1. Basic Statistics:")
        stats_df = self.basic_statistics()
        print(stats_df.head())
        
        # Target distribution
        print("\n2. Target Distribution Analysis:")
        target_counts = self.target_distribution_analysis()
        print(target_counts)
        
        # Numerical distributions
        print("\n3. Numerical Feature Distributions:")
        self.numerical_feature_distributions()
        
        # Categorical analysis
        print("\n4. Categorical Feature Analysis:")
        self.categorical_feature_analysis()
        
        # Correlation analysis
        print("\n5. Correlation Analysis:")
        corr_matrix, high_corr_pairs = self.correlation_analysis()
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        
        # Feature-target relationships
        print("\n6. Feature-Target Relationships:")
        self.feature_target_relationships()
        
        # Mutual information
        print("\n7. Mutual Information Analysis:")
        mi_scores = self.mutual_information_analysis()
        print("Top 10 features by mutual information:")
        print(mi_scores.head(10))
        
        # Statistical tests
        print("\n8. Statistical Tests:")
        test_results = self.statistical_tests()
        print(test_results)
        
        # Generate insights
        print("\n9. Key Insights:")
        insights = self.generate_insights()
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # Create dashboard
        print("\n10. Creating Summary Dashboard...")
        self.create_summary_dashboard()
        
        print("\nEDA completed successfully!")
        print("Visualizations saved to outputs/visualizations/")
        
        return {
            'statistics': stats_df,
            'target_distribution': target_counts,
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs,
            'mutual_information': mi_scores,
            'statistical_tests': test_results,
            'insights': insights
        }


def main():
    """Main function to run EDA."""
    # Initialize EDA
    eda = ExploratoryDataAnalysis('data/processed/employee_data_clean.csv')
    
    # Run complete EDA
    results = eda.run_complete_eda()
    
    # Save results
    results['statistics'].to_csv('outputs/reports/eda_statistics.csv')
    results['mutual_information'].to_csv('outputs/reports/mutual_information_scores.csv')
    results['statistical_tests'].to_csv('outputs/reports/statistical_tests.csv')
    
    # Save insights
    with open('outputs/reports/eda_insights.txt', 'w') as f:
        f.write("EDA INSIGHTS\n")
        f.write("="*50 + "\n\n")
        for i, insight in enumerate(results['insights'], 1):
            f.write(f"{i}. {insight}\n")
    
    print("\nEDA results saved to outputs/reports/")


if __name__ == "__main__":
    main()
