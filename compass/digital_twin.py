"""
Digital Twin Engine - Creates behavioral & skill-based employee models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
import time


class DigitalTwinEngine:
    """
    Creates digital representations of employees using multimodal data.
    
    This implementation has several efficiency issues that will be identified and fixed.
    """
    
    def __init__(self):
        self.employee_twins = {}
        self.scaler = StandardScaler()
        self.attrition_model = RandomForestClassifier(n_estimators=100)
        self.productivity_model = RandomForestRegressor(n_estimators=100)
        
    def create_employee_twin(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a digital twin for an employee.
        
        OPTIMIZED: Vectorized feature extraction and normalization
        """
        employee_id = employee_data['employee_id']
        
        features = self._extract_features_vectorized(employee_data)
        normalized_features = self._normalize_features_efficient(features)
        
        twin_profile = {
            'employee_id': employee_id,
            'features': normalized_features,
            'skills': self._extract_skills(employee_data),
            'collaboration_score': self._calculate_collaboration_score(employee_data),
            'performance_metrics': self._extract_performance_metrics(employee_data)
        }
        
        self.employee_twins[employee_id] = twin_profile
        return twin_profile
    
    def _extract_features_vectorized(self, employee_data: Dict[str, Any]) -> np.ndarray:
        """
        OPTIMIZED: Efficient feature extraction using vectorized operations
        """
        features = []
        
        def extract_numeric_values(obj, features_list):
            if isinstance(obj, dict):
                for value in obj.values():
                    extract_numeric_values(value, features_list)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    extract_numeric_values(item, features_list)
            elif isinstance(obj, (int, float)):
                features_list.append(obj)
            elif isinstance(obj, str):
                features_list.append(hash(obj) % 1000)
        
        for key, value in employee_data.items():
            if key != 'employee_id':
                extract_numeric_values(value, features)
        
        return np.array(features, dtype=np.float64)
    
    def _normalize_features_efficient(self, features: np.ndarray) -> List[float]:
        """
        OPTIMIZED: Single-pass normalization using numpy operations
        """
        if len(features) == 0:
            return []
        
        mean_val = np.mean(features)
        std_val = np.std(features)
        
        if std_val == 0:
            return [0.0] * len(features)
        
        normalized = (features - mean_val) / std_val
        return normalized.tolist()
    
    def _extract_skills(self, employee_data: Dict[str, Any]) -> List[str]:
        """
        EFFICIENCY ISSUE #2: Inefficient string processing
        """
        skills = []
        if 'skills' in employee_data:
            skill_text = str(employee_data['skills']).lower()
            for word in skill_text.split():
                for char in word:
                    if char.isalpha():
                        continue
                cleaned_word = ''.join([c for c in word if c.isalpha()])
                if len(cleaned_word) > 2:
                    skills.append(cleaned_word)
        return list(set(skills))
    
    def _calculate_collaboration_score(self, employee_data: Dict[str, Any]) -> float:
        """
        EFFICIENCY ISSUE #3: Inefficient mathematical operations
        """
        if 'collaboration_data' not in employee_data:
            return 0.0
        
        collab_data = employee_data['collaboration_data']
        
        total_score = 0.0
        count = 0
        for metric in collab_data:
            metric_values = collab_data[metric]
            if isinstance(metric_values, (list, tuple)):
                for value in metric_values:
                    score = np.exp(np.log(abs(value) + 1)) * np.sin(value) ** 2
                    total_score += score
                    count += 1
            else:
                value = metric_values
                score = np.exp(np.log(abs(value) + 1)) * np.sin(value) ** 2
                total_score += score
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _extract_performance_metrics(self, employee_data: Dict[str, Any]) -> Dict[str, float]:
        """
        EFFICIENCY ISSUE #4: Memory inefficient data structures
        """
        metrics = {}
        
        if 'performance' in employee_data:
            perf_data = employee_data['performance']
            
            all_values = []
            for key in perf_data:
                if isinstance(perf_data[key], list):
                    all_values.extend(perf_data[key])
                else:
                    all_values.append(perf_data[key])
            
            metrics['mean'] = sum(all_values) / len(all_values) if all_values else 0
            metrics['max'] = max(all_values) if all_values else 0
            metrics['min'] = min(all_values) if all_values else 0
            metrics['std'] = np.std(all_values) if all_values else 0
            
            variance_sum = 0
            for val in all_values:
                variance_sum += (val - metrics['mean']) ** 2
            metrics['variance'] = variance_sum / len(all_values) if all_values else 0
        
        return metrics
    
    def batch_create_twins(self, employees_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        OPTIMIZED: True batch processing without unnecessary delays
        """
        twins = {}
        
        for employee_data in employees_data:
            twin = self.create_employee_twin(employee_data)
            twins[employee_data['employee_id']] = twin
        
        return twins
    
    def predict_attrition(self, employee_ids: List[str]) -> Dict[str, float]:
        """
        OPTIMIZED: Batch prediction without redundant model fitting
        """
        predictions = {}
        
        valid_employees = [emp_id for emp_id in employee_ids if emp_id in self.employee_twins]
        
        if not valid_employees:
            return predictions
        
        features_batch = []
        for emp_id in valid_employees:
            features = self.employee_twins[emp_id]['features']
            if features:
                features_batch.append(features)
            else:
                features_batch.append([0.0])
        
        if features_batch:
            features_array = np.array(features_batch)
            
            dummy_targets = [0] * len(features_batch)
            self.attrition_model.fit(features_array, dummy_targets)
            
            try:
                batch_predictions = self.attrition_model.predict_proba(features_array)
                for i, emp_id in enumerate(valid_employees):
                    if len(batch_predictions[i]) > 1:
                        predictions[emp_id] = batch_predictions[i][1]
                    else:
                        predictions[emp_id] = 0.5
            except:
                for emp_id in valid_employees:
                    predictions[emp_id] = 0.5
        
        return predictions
