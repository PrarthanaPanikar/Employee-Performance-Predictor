"""
Employee Performance Predictor - Data Cleaning Module

This module handles data cleaning, validation, and preprocessing for the employee dataset.
It includes functions for handling missing values, outliers, data validation, and quality checks.

Author: Data Science Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    """
    Comprehensive data cleaning and validation for employee performance data.
    """
    
    def __init__(self):
        """Initialize the data cleaner with validation rules."""
        self.validation_rules = {
            'age': {'min': 22, 'max': 65},
            'experience_years': {'min': 0, 'max': 45},
            'company_tenure': {'min': 0, 'max': 30},
            'salary': {'min': 30000, 'max': 300000},
            'promotion_history': {'min': 0, 'max': 10},
            'quality_score': {'min': 1, 'max': 5},
            'peer_feedback_score': {'min': 1, 'max': 5},
            'manager_score': {'min': 1, 'max': 5},
            'on_time_delivery_rate': {'min': 0, 'max': 1},
            'training_hours': {'min': 0, 'max': 500},
            'certifications_count': {'min': 0, 'max': 20},
            'projects_completed': {'min': 0, 'max': 200},
            'attendance_rate': {'min': 0.5, 'max': 1.0},
            'sick_days': {'min': 0, 'max': 60},
            'overtime_hours': {'min': 0, 'max': 500}
        }
        
        self.categorical_columns = [
            'employee_id', 'gender', 'department', 'job_level', 
            'education_level', 'performance_band'
        ]
        
        self.numerical_columns = [
            'age', 'experience_years', 'company_tenure', 'salary',
            'promotion_history', 'quality_score', 'peer_feedback_score',
            'manager_score', 'on_time_delivery_rate', 'training_hours',
            'certifications_count', 'projects_completed', 'attendance_rate',
            'sick_days', 'overtime_hours', 'salary_per_experience',
            'training_per_year', 'projects_per_year'
        ]
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file with basic validation.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def basic_info(self, df: pd.DataFrame) -> Dict:
        """
        Generate basic information about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'duplicates': df.duplicated().sum()
        }
        
        return info
    
    def validate_data_ranges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate data against predefined ranges and fix outliers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, validation report)
        """
        df_clean = df.copy()
        validation_report = {}
        
        for column, rules in self.validation_rules.items():
            if column in df_clean.columns:
                # Count violations
                min_violations = (df_clean[column] < rules['min']).sum()
                max_violations = (df_clean[column] > rules['max']).sum()
                
                validation_report[column] = {
                    'min_violations': min_violations,
                    'max_violations': max_violations,
                    'total_violations': min_violations + max_violations
                }
                
                # Fix outliers by clipping to valid range
                df_clean[column] = df_clean[column].clip(
                    lower=rules['min'], 
                    upper=rules['max']
                )
        
        return df_clean, validation_report
    
    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values with appropriate strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, missing value report)
        """
        df_clean = df.copy()
        missing_report = {}
        
        # Check for missing values
        missing_counts = df_clean.isnull().sum()
        
        for column in df_clean.columns:
            missing_count = missing_counts[column]
            
            if missing_count > 0:
                missing_report[column] = {
                    'missing_count': missing_count,
                    'missing_percentage': (missing_count / len(df_clean)) * 100
                }
                
                # Handle missing values based on column type
                if column in self.categorical_columns:
                    # For categorical, use mode
                    mode_value = df_clean[column].mode()[0] if not df_clean[column].mode().empty else 'Unknown'
                    df_clean[column] = df_clean[column].fillna(mode_value)
                    missing_report[column]['strategy'] = 'mode'
                    
                elif column in self.numerical_columns:
                    # For numerical, use median (more robust to outliers)
                    median_value = df_clean[column].median()
                    df_clean[column] = df_clean[column].fillna(median_value)
                    missing_report[column]['strategy'] = 'median'
        
        return df_clean, missing_report
    
    def detect_and_handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and handle duplicate records.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, duplicate report)
        """
        df_clean = df.copy()
        
        # Check for exact duplicates
        exact_duplicates = df_clean.duplicated().sum()
        
        # Check for duplicate employee IDs (should be unique)
        duplicate_ids = df_clean['employee_id'].duplicated().sum()
        
        duplicate_report = {
            'exact_duplicates': exact_duplicates,
            'duplicate_employee_ids': duplicate_ids,
            'total_duplicates': exact_duplicates + duplicate_ids
        }
        
        # Remove exact duplicates
        if exact_duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            print(f"Removed {exact_duplicates} exact duplicate rows")
        
        # Handle duplicate employee IDs (keep the first occurrence)
        if duplicate_ids > 0:
            before_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['employee_id'], keep='first')
            after_count = len(df_clean)
            removed_ids = before_count - after_count
            print(f"Removed {removed_ids} duplicate employee IDs")
        
        return df_clean, duplicate_report
    
    def validate_data_consistency(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate logical consistency in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cleaned DataFrame, consistency report)
        """
        df_clean = df.copy()
        consistency_report = {}
        
        # Consistency check 1: experience_years >= company_tenure
        exp_violations = (df_clean['experience_years'] < df_clean['company_tenure']).sum()
        consistency_report['experience_vs_tenure'] = {
            'violations': exp_violations,
            'description': 'Experience years should be >= company tenure'
        }
        
        if exp_violations > 0:
            # Fix by setting experience_years = max(experience_years, company_tenure)
            df_clean['experience_years'] = np.maximum(
                df_clean['experience_years'], 
                df_clean['company_tenure']
            )
        
        # Consistency check 2: reasonable salary ranges for job levels
        salary_issues = 0
        for level in ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']:
            level_data = df_clean[df_clean['job_level'] == level]
            
            if level == 'Junior' and len(level_data) > 0:
                issues = ((level_data['salary'] < 40000) | (level_data['salary'] > 100000)).sum()
            elif level == 'Mid' and len(level_data) > 0:
                issues = ((level_data['salary'] < 60000) | (level_data['salary'] > 150000)).sum()
            elif level == 'Senior' and len(level_data) > 0:
                issues = ((level_data['salary'] < 80000) | (level_data['salary'] > 200000)).sum()
            elif level == 'Lead' and len(level_data) > 0:
                issues = ((level_data['salary'] < 100000) | (level_data['salary'] > 250000)).sum()
            elif level == 'Manager' and len(level_data) > 0:
                issues = ((level_data['salary'] < 120000) | (level_data['salary'] > 300000)).sum()
            else:
                issues = 0
            
            salary_issues += issues
        
        consistency_report['salary_job_level'] = {
            'violations': salary_issues,
            'description': 'Salary should be appropriate for job level'
        }
        
        # Consistency check 3: performance scores should be consistent
        score_violations = 0
        for score_col in ['quality_score', 'peer_feedback_score', 'manager_score']:
            if score_col in df_clean.columns:
                violations = ((df_clean[score_col] < 1) | (df_clean[score_col] > 5)).sum()
                score_violations += violations
        
        consistency_report['performance_scores'] = {
            'violations': score_violations,
            'description': 'Performance scores should be between 1 and 5'
        }
        
        return df_clean, consistency_report
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_clean = df.copy()
        
        # Experience ratio
        df_clean['experience_ratio'] = df_clean['company_tenure'] / (df_clean['experience_years'] + 1)
        
        # Salary per year of tenure
        df_clean['salary_per_tenure'] = df_clean['salary'] / (df_clean['company_tenure'] + 1)
        
        # Performance composite score
        if all(col in df_clean.columns for col in ['quality_score', 'peer_feedback_score', 'manager_score']):
            df_clean['performance_composite'] = (
                df_clean['quality_score'] * 0.4 +
                df_clean['peer_feedback_score'] * 0.3 +
                df_clean['manager_score'] * 0.3
            )
        
        # Engagement score (combination of training, attendance, and overtime)
        df_clean['engagement_score'] = (
            (df_clean['training_hours'] / 100) * 0.3 +
            (df_clean['attendance_rate']) * 0.4 +
            (df_clean['overtime_hours'] / 100) * 0.3
        )
        
        # Productivity score (projects per experience)
        df_clean['productivity_score'] = df_clean['projects_completed'] / (df_clean['experience_years'] + 1)
        
        print(f"Created {5} engineered features")
        
        return df_clean
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, 
                              cleaned_df: pd.DataFrame,
                              validation_report: Dict,
                              missing_report: Dict,
                              duplicate_report: Dict,
                              consistency_report: Dict) -> Dict:
        """
        Generate comprehensive cleaning report.
        
        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            validation_report: Range validation report
            missing_report: Missing value report
            duplicate_report: Duplicate report
            consistency_report: Consistency validation report
            
        Returns:
            Comprehensive cleaning report
        """
        report = {
            'original_shape': original_df.shape,
            'cleaned_shape': cleaned_df.shape,
            'rows_removed': original_df.shape[0] - cleaned_df.shape[0],
            'columns_added': cleaned_df.shape[1] - original_df.shape[1],
            
            'validation_issues': {
                'total_range_violations': sum(
                    report['total_violations'] for report in validation_report.values()
                ),
                'columns_with_violations': len(validation_report)
            },
            
            'missing_value_issues': {
                'total_missing_values': sum(
                    report['missing_count'] for report in missing_report.values()
                ),
                'columns_with_missing': len(missing_report)
            },
            
            'duplicate_issues': duplicate_report,
            
            'consistency_issues': {
                'total_consistency_violations': sum(
                    report['violations'] for report in consistency_report.values()
                ),
                'consistency_checks': len(consistency_report)
            },
            
            'data_quality_score': self._calculate_quality_score(
                original_df, cleaned_df, validation_report, missing_report, 
                duplicate_report, consistency_report
            )
        }
        
        return report
    
    def _calculate_quality_score(self, original_df: pd.DataFrame, 
                               cleaned_df: pd.DataFrame,
                               validation_report: Dict,
                               missing_report: Dict,
                               duplicate_report: Dict,
                               consistency_report: Dict) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Args:
            original_df: Original DataFrame
            cleaned_df: Cleaned DataFrame
            validation_report: Range validation report
            missing_report: Missing value report
            duplicate_report: Duplicate report
            consistency_report: Consistency validation report
            
        Returns:
            Quality score between 0 and 100
        """
        total_cells = original_df.shape[0] * original_df.shape[1]
        
        # Penalties
        missing_penalty = sum(report['missing_count'] for report in missing_report.values())
        duplicate_penalty = duplicate_report['exact_duplicates'] + duplicate_report['duplicate_employee_ids']
        range_penalty = sum(report['total_violations'] for report in validation_report.values())
        consistency_penalty = sum(report['violations'] for report in consistency_report.values())
        
        total_issues = missing_penalty + duplicate_penalty + range_penalty + consistency_penalty
        
        # Quality score (100 - percentage of problematic cells)
        quality_score = max(0, 100 - (total_issues / total_cells * 100))
        
        return round(quality_score, 2)
    
    def clean_data(self, filepath: str, save_cleaned: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data cleaning pipeline.
        
        Args:
            filepath: Path to the raw data file
            save_cleaned: Whether to save the cleaned data
            
        Returns:
            Tuple of (cleaned DataFrame, cleaning report)
        """
        print("Starting data cleaning process...")
        
        # Load data
        df = self.load_data(filepath)
        original_df = df.copy()
        
        print(f"Loaded data: {df.shape}")
        
        # Step 1: Basic info
        basic_info = self.basic_info(df)
        print(f"Missing values: {sum(basic_info['missing_values'].values())}")
        print(f"Duplicates: {basic_info['duplicates']}")
        
        # Step 2: Validate data ranges
        df, validation_report = self.validate_data_ranges(df)
        print(f"Fixed {sum(report['total_violations'] for report in validation_report.values())} range violations")
        
        # Step 3: Handle missing values
        df, missing_report = self.handle_missing_values(df)
        if missing_report:
            print(f"Handled missing values in {len(missing_report)} columns")
        else:
            print("No missing values found")
        
        # Step 4: Handle duplicates
        df, duplicate_report = self.detect_and_handle_duplicates(df)
        print(f"Removed {duplicate_report['total_duplicates']} duplicate records")
        
        # Step 5: Validate consistency
        df, consistency_report = self.validate_data_consistency(df)
        print(f"Fixed {sum(report['violations'] for report in consistency_report.values())} consistency issues")
        
        # Step 6: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 7: Generate cleaning report
        cleaning_report = self.generate_cleaning_report(
            original_df, df, validation_report, missing_report, 
            duplicate_report, consistency_report
        )
        
        print(f"\nCleaning completed!")
        print(f"Final shape: {df.shape}")
        print(f"Data quality score: {cleaning_report['data_quality_score']}/100")
        
        # Save cleaned data
        if save_cleaned:
            output_path = 'data/processed/employee_data_clean.csv'
            df.to_csv(output_path, index=False)
            print(f"Cleaned data saved to {output_path}")
        
        return df, cleaning_report


def main():
    """Main function to run data cleaning."""
    cleaner = DataCleaner()
    
    # Clean the data
    cleaned_data, report = cleaner.clean_data('data/raw/employee_data_raw.csv')
    
    # Print summary
    print("\n" + "="*50)
    print("DATA CLEANING SUMMARY")
    print("="*50)
    print(f"Original shape: {report['original_shape']}")
    print(f"Cleaned shape: {report['cleaned_shape']}")
    print(f"Rows removed: {report['rows_removed']}")
    print(f"Columns added: {report['columns_added']}")
    print(f"Data quality score: {report['data_quality_score']}/100")
    print(f"Total issues fixed: {sum([
        report['validation_issues']['total_range_violations'],
        report['missing_value_issues']['total_missing_values'],
        report['duplicate_issues']['total_duplicates'],
        report['consistency_issues']['total_consistency_violations']
    ])}")


if __name__ == "__main__":
    main()
