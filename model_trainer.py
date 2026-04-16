"""
Employee Performance Predictor - Model Training Module

This module implements the complete machine learning pipeline including:
- Data preprocessing and feature engineering
- Model training with multiple algorithms
- Hyperparameter tuning
- Cross-validation
- Model comparison and selection

Author: Data Science Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import joblib
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive machine learning pipeline for employee performance prediction.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.model_results = {}
        self.feature_names = None
        self.target_encoder = None
        
        # Define feature categories
        self.numerical_features = [
            'age', 'experience_years', 'company_tenure', 'salary', 'promotion_history',
            'quality_score', 'peer_feedback_score', 'manager_score', 'on_time_delivery_rate',
            'training_hours', 'certifications_count', 'projects_completed', 'attendance_rate',
            'sick_days', 'overtime_hours', 'salary_per_experience', 'training_per_year',
            'projects_per_year', 'experience_ratio', 'salary_per_tenure', 'performance_composite',
            'engagement_score', 'productivity_score'
        ]
        
        self.categorical_features = [
            'department', 'job_level', 'education_level', 'gender'
        ]
        
        # Target column
        self.target_column = 'performance_band'
        
        # Define models to train
        self.model_definitions = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__class_weight': ['balanced']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 15, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__class_weight': ['balanced']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.1, 0.05],
                    'classifier__max_depth': [3, 5]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=random_state, eval_metric='mlogloss'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.1, 0.05],
                    'classifier__max_depth': [3, 5],
                    'classifier__subsample': [0.8, 1.0]
                }
            }
        }
    
    def load_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training.
        
        Args:
            filepath: Path to the cleaned dataset
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(columns=[self.target_column, 'employee_id'])
        y = df[self.target_column]
        
        # Filter available features
        available_numerical = [col for col in self.numerical_features if col in X.columns]
        available_categorical = [col for col in self.categorical_features if col in X.columns]
        
        self.feature_names = available_numerical + available_categorical
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Available numerical features: {len(available_numerical)}")
        print(f"Available categorical features: {len(available_categorical)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numerical and categorical features.
        
        Returns:
            ColumnTransformer with preprocessing steps
        """
        # Numerical preprocessing
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, 
                 [col for col in self.numerical_features if col in self.feature_names]),
                ('cat', categorical_transformer, 
                 [col for col in self.categorical_features if col in self.feature_names])
            ]
        )
        
        return preprocessor
    
    def encode_target(self, y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
        """
        Encode target variable.
        
        Args:
            y: Target Series
            
        Returns:
            Tuple of (encoded target array, LabelEncoder)
        """
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        self.target_encoder = encoder
        
        print(f"Target classes: {list(encoder.classes_)}")
        print(f"Encoded target shape: {y_encoded.shape}")
        
        return y_encoded, encoder
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray, 
                   test_size: float = 0.2) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features DataFrame
            y: Target array
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training results
        """
        print(f"\nTraining {model_name}...")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()
        
        # Get model definition
        model_def = self.model_definitions[model_name]
        
        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_def['model'])
        ])
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, 
            model_def['params'], 
            cv=cv, 
            scoring='f1_macro',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        
        # Store results
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        print(f"Best params: {grid_search.best_params_}")
        
        return results
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict:
        """
        Train all models and compare results.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary with all model results
        """
        print("Starting model training pipeline...")
        
        all_results = {}
        
        for model_name in self.model_definitions.keys():
            try:
                results = self.train_single_model(model_name, X_train, y_train)
                all_results[model_name] = results
                self.models[model_name] = results['best_model']
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model
        if all_results:
            best_model_name = max(all_results.keys(), 
                                key=lambda x: all_results[x]['best_score'])
            self.best_model = all_results[best_model_name]['best_model']
            
            print(f"\nBest model: {best_model_name}")
            print(f"Best CV score: {all_results[best_model_name]['best_score']:.4f}")
        
        self.model_results = all_results
        return all_results
    
    def evaluate_model(self, model: Pipeline, X_test: pd.DataFrame, 
                      y_test: np.ndarray, model_name: str) -> Dict:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = None
        
        # Get probabilities if available
        if hasattr(model.named_steps['classifier'], 'predict_proba'):
            y_proba = model.named_steps['classifier'].predict_proba(
                model.named_steps['preprocessor'].transform(X_test)
            )
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                         target_names=self.target_encoder.classes_,
                                         output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist() if y_proba is not None else None
        }
        
        print(f"\n{model_name} Test Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1:.4f}")
        print(f"Precision (Macro): {precision:.4f}")
        print(f"Recall (Macro): {recall:.4f}")
        
        return results
    
    def evaluate_all_models(self, X_test: pd.DataFrame, 
                           y_test: np.ndarray) -> Dict:
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with all evaluation results
        """
        print("\nEvaluating models on test data...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                results = self.evaluate_model(model, X_test, y_test, model_name)
                evaluation_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Create comparison table
        if evaluation_results:
            comparison_data = []
            for model_name, results in evaluation_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'F1-Macro': results['f1_macro'],
                    'Precision-Macro': results['precision_macro'],
                    'Recall-Macro': results['recall_macro']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('F1-Macro', ascending=False)
            
            print("\nModel Comparison:")
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv('outputs/reports/model_comparison.csv', index=False)
        
        return evaluation_results
    
    def get_feature_importance(self, model_name: str) -> Dict:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with feature importance
        """
        model = self.models[model_name]
        
        if model_name == 'logistic_regression':
            # Get coefficients for logistic regression
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            
            # Get feature names after preprocessing
            feature_names = []
            # Numerical features
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(num_features)
            # Categorical features
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_features)
            
            # Get coefficients (for multi-class, take mean absolute)
            if len(classifier.coef_) > 1:
                # Multi-class case
                importance = np.mean(np.abs(classifier.coef_), axis=0)
            else:
                # Binary case
                importance = np.abs(classifier.coef_[0])
            
        elif model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
            # Get feature importances for tree-based models
            classifier = model.named_steps['classifier']
            importance = classifier.feature_importances_
            
            # Get feature names after preprocessing
            preprocessor = model.named_steps['preprocessor']
            feature_names = []
            # Numerical features
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(num_features)
            # Categorical features
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_features)
        
        else:
            return {}
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                     key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def save_models(self, save_path: str = 'models/trained/'):
        """
        Save all trained models.
        
        Args:
            save_path: Path to save models
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save best model
        if self.best_model:
            joblib.dump(self.best_model, f'{save_path}best_model.pkl')
            print(f"Best model saved to {save_path}best_model.pkl")
        
        # Save all models
        for model_name, model in self.models.items():
            joblib.dump(model, f'{save_path}{model_name}_model.pkl')
        
        # Save target encoder
        if self.target_encoder:
            joblib.dump(self.target_encoder, f'{save_path}target_encoder.pkl')
        
        # Save model results
        if self.model_results:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model_name, results in self.model_results.items():
                json_results[model_name] = {
                    'model_name': results['model_name'],
                    'best_params': results['best_params'],
                    'best_score': float(results['best_score'])
                }
            
            with open(f'{save_path}model_results.json', 'w') as f:
                json.dump(json_results, f, indent=2)
        
        print(f"All models saved to {save_path}")
    
    def train_pipeline(self, data_path: str, save_models: bool = True) -> Dict:
        """
        Complete training pipeline.
        
        Args:
            data_path: Path to the training data
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with complete training results
        """
        print("Starting complete training pipeline...")
        
        # Load data
        X, y = self.load_data(data_path)
        
        # Encode target
        y_encoded, encoder = self.encode_target(y)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y_encoded)
        
        # Train models
        training_results = self.train_all_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = self.evaluate_all_models(X_test, y_test)
        
        # Get feature importance for best model
        if self.best_model:
            # Find best model name
            best_model_name = None
            best_score = 0
            for name, results in training_results.items():
                if results['best_score'] > best_score:
                    best_score = results['best_score']
                    best_model_name = name
            
            if best_model_name:
                feature_importance = self.get_feature_importance(best_model_name)
                print(f"\nTop 10 Important Features for {best_model_name}:")
                for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                    print(f"{i+1:2d}. {feature}: {importance:.4f}")
        
        # Save models
        if save_models:
            self.save_models()
        
        # Generate training report
        report = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'data_shape': X.shape,
            'feature_names': self.feature_names,
            'target_classes': list(encoder.classes_)
        }
        
        print("\nTraining pipeline completed successfully!")
        
        return report


def main():
    """Main function to run the training pipeline."""
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Run training pipeline
    results = trainer.train_pipeline('data/processed/employee_data_clean.csv')
    
    print("\nTraining completed!")
    print(f"Models trained: {len(trainer.models)}")
    print(f"Best model CV score: {max([r['best_score'] for r in results['training_results'].values()]):.4f}")


if __name__ == "__main__":
    main()
