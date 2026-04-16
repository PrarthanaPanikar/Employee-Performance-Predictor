"""
Employee Performance Predictor - Synthetic Data Generator

This module generates realistic synthetic HR data for training machine learning models.
The data simulates employee characteristics and performance metrics that would
typically be found in a medium to large technology company.

Author: Data Science Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EmployeeDataGenerator:
    """
    Generate synthetic employee data with realistic distributions and correlations.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Company parameters
        self.departments = [
            'Engineering', 'Sales', 'Marketing', 'Human Resources', 
            'Finance', 'Product', 'Operations', 'Customer Support'
        ]
        
        self.job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
        self.education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
        
        # Performance calculation weights
        self.performance_weights = {
            'quality_score': 0.25,
            'peer_feedback_score': 0.20,
            'manager_score': 0.25,
            'on_time_delivery_rate': 0.15,
            'training_hours': 0.10,
            'attendance_rate': 0.05
        }
    
    def generate_employee_demographics(self, n_employees: int) -> pd.DataFrame:
        """
        Generate basic employee demographic information.
        
        Args:
            n_employees: Number of employees to generate
            
        Returns:
            DataFrame with employee demographics
        """
        employees = []
        
        for i in range(n_employees):
            emp_id = f"EMP_{str(i+1).zfill(4)}"
            
            # Age distribution (realistic for tech companies)
            age = np.random.normal(35, 8)
            age = np.clip(age, 22, 65)
            
            # Gender distribution (approximately balanced)
            gender = np.random.choice(['Male', 'Female'], p=[0.52, 0.48])
            
            # Department distribution (Engineering-heavy tech company)
            dept_probs = [0.35, 0.15, 0.12, 0.08, 0.10, 0.08, 0.07, 0.05]
            department = np.random.choice(self.departments, p=dept_probs)
            
            # Job level distribution (pyramid structure)
            level_probs = [0.40, 0.30, 0.20, 0.07, 0.03]
            job_level = np.random.choice(self.job_levels, p=level_probs)
            
            # Education level distribution
            edu_probs = [0.05, 0.60, 0.30, 0.05]
            education = np.random.choice(self.education_levels, p=edu_probs)
            
            # Experience and tenure (correlated with age and job level)
            if job_level == 'Junior':
                experience_years = max(0, np.random.normal(2, 1.5))
                company_tenure = max(0, np.random.normal(1, 0.8))
            elif job_level == 'Mid':
                experience_years = max(1, np.random.normal(5, 2))
                company_tenure = max(0.5, np.random.normal(3, 1.5))
            elif job_level == 'Senior':
                experience_years = max(3, np.random.normal(8, 3))
                company_tenure = max(1, np.random.normal(5, 2))
            elif job_level == 'Lead':
                experience_years = max(5, np.random.normal(12, 4))
                company_tenure = max(2, np.random.normal(7, 3))
            else:  # Manager
                experience_years = max(8, np.random.normal(15, 5))
                company_tenure = max(3, np.random.normal(8, 4))
            
            # Salary based on job level, experience, and department
            base_salaries = {
                'Junior': 60000, 'Mid': 85000, 'Senior': 120000, 
                'Lead': 150000, 'Manager': 180000
            }
            
            dept_multipliers = {
                'Engineering': 1.2, 'Product': 1.15, 'Finance': 1.1,
                'Sales': 1.05, 'Marketing': 1.0, 'Operations': 0.95,
                'Human Resources': 0.9, 'Customer Support': 0.85
            }
            
            base_salary = base_salaries[job_level]
            dept_multiplier = dept_multipliers[department]
            experience_bonus = experience_years * 1000
            
            salary = (base_salary * dept_multiplier + experience_bonus + 
                     np.random.normal(0, 5000))
            salary = max(35000, min(250000, salary))
            
            # Promotion history
            if company_tenure < 1:
                promotions = 0
            elif company_tenure < 3:
                promotions = np.random.choice([0, 1], p=[0.7, 0.3])
            elif company_tenure < 6:
                promotions = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
            else:
                promotions = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            
            employees.append({
                'employee_id': emp_id,
                'age': int(age),
                'gender': gender,
                'department': department,
                'job_level': job_level,
                'education_level': education,
                'experience_years': round(experience_years, 1),
                'company_tenure': round(company_tenure, 1),
                'salary': round(salary, 0),
                'promotion_history': promotions
            })
        
        return pd.DataFrame(employees)
    
    def generate_performance_metrics(self, demographics: pd.DataFrame) -> pd.DataFrame:
        """
        Generate performance metrics based on employee characteristics.
        
        Args:
            demographics: DataFrame with employee demographics
            
        Returns:
            DataFrame with performance metrics
        """
        performance_data = []
        
        for _, emp in demographics.iterrows():
            # Base performance influenced by demographics
            base_performance = 3.0  # Middle of 1-5 scale
            
            # Job level influence
            level_impacts = {
                'Junior': -0.3, 'Mid': 0.0, 'Senior': 0.3, 
                'Lead': 0.5, 'Manager': 0.4
            }
            base_performance += level_impacts[emp['job_level']]
            
            # Experience influence
            exp_impact = min(0.5, emp['experience_years'] * 0.02)
            base_performance += exp_impact
            
            # Education influence
            edu_impacts = {
                'High School': -0.2, 'Bachelor': 0.0, 
                'Master': 0.2, 'PhD': 0.3
            }
            base_performance += edu_impacts[emp['education_level']]
            
            # Add random variation
            random_factor = np.random.normal(0, 0.3)
            base_performance += random_factor
            
            # Generate individual metrics with correlations
            quality_score = np.clip(base_performance + np.random.normal(0, 0.4), 1, 5)
            
            # Peer feedback tends to correlate with quality
            peer_feedback_score = np.clip(
                quality_score * 0.8 + np.random.normal(0, 0.3), 1, 5
            )
            
            # Manager score (slightly different perspective)
            manager_score = np.clip(
                base_performance + np.random.normal(0, 0.5), 1, 5
            )
            
            # On-time delivery rate (influenced by experience and job level)
            base_delivery = 0.85
            if emp['job_level'] == 'Junior':
                base_delivery -= 0.1
            elif emp['job_level'] in ['Senior', 'Lead']:
                base_delivery += 0.05
            
            on_time_delivery_rate = np.clip(
                base_delivery + np.random.normal(0, 0.1), 0.5, 1.0
            )
            
            # Training hours (varies by department and career stage)
            base_training = 20
            if emp['job_level'] == 'Junior':
                base_training += 15
            elif emp['department'] in ['Engineering', 'Product']:
                base_training += 10
            
            training_hours = max(0, base_training + np.random.normal(0, 15))
            
            # Certifications (correlated with training and ambition)
            cert_prob = min(0.8, training_hours / 50)
            if emp['education_level'] in ['Master', 'PhD']:
                cert_prob += 0.2
            
            certifications_count = np.random.choice(
                [0, 1, 2, 3, 4, 5], 
                p=[1-cert_prob, cert_prob*0.4, cert_prob*0.3, 
                   cert_prob*0.2, cert_prob*0.08, cert_prob*0.02]
            )
            
            # Projects completed (based on tenure and role)
            projects_per_year = 10 if emp['department'] == 'Engineering' else 8
            if emp['job_level'] == 'Manager':
                projects_per_year = 15  # Overseeing more projects
            
            projects_completed = int(
                projects_per_year * emp['company_tenure'] + 
                np.random.normal(0, 3)
            )
            projects_completed = max(0, projects_completed)
            
            # Attendance rate (generally high with some variation)
            attendance_rate = np.clip(
                0.95 + np.random.normal(0, 0.03), 0.7, 1.0
            )
            
            # Sick days (correlated with attendance)
            sick_days = int((1 - attendance_rate) * 250 + np.random.normal(0, 2))
            sick_days = max(0, min(30, sick_days))
            
            # Overtime hours (varies by department and role)
            base_overtime = 20
            if emp['department'] in ['Engineering', 'Sales']:
                base_overtime += 15
            if emp['job_level'] in ['Senior', 'Lead', 'Manager']:
                base_overtime += 10
            
            overtime_hours = max(0, base_overtime + np.random.normal(0, 15))
            
            performance_data.append({
                'employee_id': emp['employee_id'],
                'quality_score': round(quality_score, 2),
                'peer_feedback_score': round(peer_feedback_score, 2),
                'manager_score': round(manager_score, 2),
                'on_time_delivery_rate': round(on_time_delivery_rate, 3),
                'training_hours': round(training_hours, 1),
                'certifications_count': certifications_count,
                'projects_completed': projects_completed,
                'attendance_rate': round(attendance_rate, 3),
                'sick_days': sick_days,
                'overtime_hours': round(overtime_hours, 1)
            })
        
        return pd.DataFrame(performance_data)
    
    def calculate_performance_band(self, demographics: pd.DataFrame, 
                                 performance: pd.DataFrame) -> pd.Series:
        """
        Calculate performance bands based on weighted performance metrics.
        
        Args:
            demographics: Employee demographics
            performance: Performance metrics
            
        Returns:
            Series with performance bands (High/Medium/Low)
        """
        bands = []
        
        for _, perf in performance.iterrows():
            # Calculate weighted performance score
            weighted_score = (
                perf['quality_score'] * self.performance_weights['quality_score'] +
                perf['peer_feedback_score'] * self.performance_weights['peer_feedback_score'] +
                perf['manager_score'] * self.performance_weights['manager_score'] +
                perf['on_time_delivery_rate'] * 5 * self.performance_weights['on_time_delivery_rate'] +
                (perf['training_hours'] / 40) * 5 * self.performance_weights['training_hours'] +
                perf['attendance_rate'] * 5 * self.performance_weights['attendance_rate']
            )
            
            # Add some randomness for realistic variation
            weighted_score += np.random.normal(0, 0.2)
            weighted_score = np.clip(weighted_score, 1, 5)
            
            # Determine performance band
            if weighted_score >= 4.0:
                band = 'High'
            elif weighted_score >= 2.5:
                band = 'Medium'
            else:
                band = 'Low'
            
            bands.append(band)
        
        return pd.Series(bands)
    
    def generate_dataset(self, n_employees: int = 1000) -> pd.DataFrame:
        """
        Generate complete synthetic employee dataset.
        
        Args:
            n_employees: Number of employees to generate
            
        Returns:
            Complete DataFrame with all employee data
        """
        print(f"Generating synthetic dataset with {n_employees} employees...")
        
        # Generate demographics
        demographics = self.generate_employee_demographics(n_employees)
        print("Generated employee demographics")
        
        # Generate performance metrics
        performance = self.generate_performance_metrics(demographics)
        print("Generated performance metrics")
        
        # Calculate performance bands
        performance['performance_band'] = self.calculate_performance_band(
            demographics, performance
        )
        print("Calculated performance bands")
        
        # Merge datasets
        full_dataset = demographics.merge(performance, on='employee_id')
        
        # Add some additional derived features
        full_dataset['salary_per_experience'] = (
            full_dataset['salary'] / (full_dataset['experience_years'] + 1)
        )
        
        full_dataset['training_per_year'] = (
            full_dataset['training_hours'] / (full_dataset['company_tenure'] + 0.5)
        )
        
        full_dataset['projects_per_year'] = (
            full_dataset['projects_completed'] / (full_dataset['company_tenure'] + 0.5)
        )
        
        # Display dataset statistics
        print("\nDataset Statistics:")
        print(f"Total employees: {len(full_dataset)}")
        print(f"Performance distribution:")
        print(full_dataset['performance_band'].value_counts())
        print(f"\nDepartment distribution:")
        print(full_dataset['department'].value_counts())
        print(f"\nJob level distribution:")
        print(full_dataset['job_level'].value_counts())
        
        return full_dataset
    
    def save_dataset(self, dataset: pd.DataFrame, filepath: str):
        """
        Save the generated dataset to CSV file.
        
        Args:
            dataset: DataFrame to save
            filepath: Path to save the CSV file
        """
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
    
    def generate_data_dictionary(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a data dictionary for the dataset.
        
        Args:
            dataset: The generated dataset
            
        Returns:
            DataFrame with data dictionary information
        """
        dictionary = []
        
        for column in dataset.columns:
            dtype = str(dataset[column].dtype)
            unique_count = dataset[column].nunique()
            
            if dtype == 'object':
                if unique_count < 20:
                    sample_values = str(dataset[column].unique()[:10].tolist())
                else:
                    sample_values = f"Too many unique values ({unique_count})"
            else:
                # Check if column is numeric before formatting
                if pd.api.types.is_numeric_dtype(dataset[column]):
                    sample_values = f"Min: {dataset[column].min():.2f}, Max: {dataset[column].max():.2f}"
                else:
                    sample_values = str(dataset[column].unique()[:5].tolist())
            
            description = self._get_column_description(column)
            
            dictionary.append({
                'Column_Name': column,
                'Data_Type': dtype,
                'Unique_Values': unique_count,
                'Sample_Values': sample_values,
                'Description': description
            })
        
        return pd.DataFrame(dictionary)
    
    def _get_column_description(self, column: str) -> str:
        """Get description for each column."""
        descriptions = {
            'employee_id': 'Unique identifier for each employee',
            'age': 'Employee age in years',
            'gender': 'Employee gender (for fairness audit only)',
            'department': 'Department where employee works',
            'job_level': 'Current job level/position',
            'education_level': 'Highest education level achieved',
            'experience_years': 'Total professional experience in years',
            'company_tenure': 'Years employed at current company',
            'salary': 'Annual salary in USD',
            'promotion_history': 'Number of promotions received',
            'quality_score': 'Average quality rating (1-5 scale)',
            'peer_feedback_score': 'Average peer feedback rating (1-5 scale)',
            'manager_score': 'Manager performance rating (1-5 scale)',
            'on_time_delivery_rate': 'Percentage of tasks completed on time',
            'training_hours': 'Training hours completed in last 6 months',
            'certifications_count': 'Number of professional certifications',
            'projects_completed': 'Total projects completed',
            'attendance_rate': 'Attendance percentage',
            'sick_days': 'Sick days taken in last 6 months',
            'overtime_hours': 'Overtime hours worked in last 6 months',
            'performance_band': 'Performance category (High/Medium/Low)',
            'salary_per_experience': 'Salary divided by years of experience',
            'training_per_year': 'Training hours per year of tenure',
            'projects_per_year': 'Projects completed per year of tenure'
        }
        return descriptions.get(column, 'No description available')


def main():
    """Main function to generate and save the synthetic dataset."""
    # Initialize generator
    generator = EmployeeDataGenerator(random_state=42)
    
    # Generate dataset
    dataset = generator.generate_dataset(n_employees=1000)
    
    # Save dataset
    generator.save_dataset(dataset, 'data/raw/employee_data_raw.csv')
    
    # Generate and save data dictionary
    data_dict = generator.generate_data_dictionary(dataset)
    data_dict.to_csv('data/raw/data_dictionary.csv', index=False)
    
    print("\nData generation completed successfully!")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Files saved:")
    print("  - data/raw/employee_data_raw.csv")
    print("  - data/raw/data_dictionary.csv")


if __name__ == "__main__":
    main()
