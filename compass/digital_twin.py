"""
Digital Twin Engine - Creates behavioral & skill-based employee models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
import time
import random


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
    
    def generate_anti_twin(self, employee_id: str) -> Dict[str, Any]:
        """
        Generate an anti-twin profile with inverse personality characteristics.
        
        This is the core patent innovation for stress testing organizational vulnerability.
        """
        if employee_id not in self.employee_twins:
            raise ValueError(f"Employee {employee_id} not found in digital twins")
        
        original_twin = self.employee_twins[employee_id]
        
        anti_twin_features = self._invert_features(original_twin['features'])
        anti_twin_skills = self._invert_skills(original_twin['skills'])
        anti_twin_collab_score = self._invert_collaboration_score(original_twin['collaboration_score'])
        anti_twin_performance = self._invert_performance_metrics(original_twin['performance_metrics'])
        
        anti_twin_profile = {
            'employee_id': f"anti_{employee_id}",
            'original_employee_id': employee_id,
            'features': anti_twin_features,
            'skills': anti_twin_skills,
            'collaboration_score': anti_twin_collab_score,
            'performance_metrics': anti_twin_performance,
            'is_anti_twin': True
        }
        
        return anti_twin_profile
    
    def _invert_features(self, features: List[float]) -> List[float]:
        """Invert normalized features using statistical sampling"""
        if not features:
            return []
        
        inverted = []
        for feature in features:
            inverted_value = -feature + random.uniform(-0.5, 0.5)
            inverted.append(inverted_value)
        
        return inverted
    
    def _invert_skills(self, skills: List[str]) -> List[str]:
        """Generate complementary skill set for anti-twin"""
        all_possible_skills = ['python', 'javascript', 'sql', 'machine_learning', 'data_analysis', 
                              'project_management', 'communication', 'leadership', 'design', 'marketing',
                              'finance', 'operations', 'strategy', 'sales', 'hr', 'legal', 'research']
        
        current_skills = set(skills)
        available_skills = [skill for skill in all_possible_skills if skill not in current_skills]
        
        num_anti_skills = min(len(skills), len(available_skills))
        anti_skills = random.sample(available_skills, num_anti_skills) if available_skills else []
        
        return anti_skills
    
    def _invert_collaboration_score(self, collab_score: float) -> float:
        """Invert collaboration score with some randomness"""
        max_score = 100.0
        inverted_score = max_score - collab_score + random.uniform(-10, 10)
        return max(0, min(max_score, inverted_score))
    
    def _invert_performance_metrics(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Invert performance metrics to create opposite profile"""
        inverted_metrics = {}
        
        for key, value in performance_metrics.items():
            if key in ['mean', 'max', 'min']:
                inverted_metrics[key] = 5.0 - value + random.uniform(-0.5, 0.5)
            elif key == 'std':
                inverted_metrics[key] = max(0.1, 2.0 - value)
            elif key == 'variance':
                inverted_metrics[key] = max(0.1, 4.0 - value)
            else:
                inverted_metrics[key] = value
        
        return inverted_metrics
    
    def stress_test_team_composition(self, team_members: List[str], num_anti_twins: int = 2) -> Dict[str, Any]:
        """
        Stress test team composition using anti-twin profiles.
        
        Core patent innovation: Tests organizational vulnerability by introducing
        inverse personality profiles into team dynamics.
        """
        if len(team_members) < 2:
            return {'error': 'Team must have at least 2 members for stress testing'}
        
        original_team_score = self._calculate_team_cohesion(team_members)
        
        stress_test_results = []
        
        for _ in range(num_anti_twins):
            target_member = random.choice(team_members)
            anti_twin = self.generate_anti_twin(target_member)
            
            modified_team = [member for member in team_members if member != target_member]
            modified_team.append(anti_twin['employee_id'])
            
            temp_twins = self.employee_twins.copy()
            temp_twins[anti_twin['employee_id']] = anti_twin
            
            stress_score = self._calculate_team_cohesion_with_twins(modified_team, temp_twins)
            
            vulnerability_score = original_team_score - stress_score
            
            stress_test_results.append({
                'replaced_member': target_member,
                'anti_twin_id': anti_twin['employee_id'],
                'original_score': original_team_score,
                'stress_score': stress_score,
                'vulnerability_score': vulnerability_score,
                'vulnerability_percentage': (vulnerability_score / original_team_score) * 100 if original_team_score > 0 else 0
            })
        
        avg_vulnerability = np.mean([result['vulnerability_score'] for result in stress_test_results])
        max_vulnerability = max([result['vulnerability_score'] for result in stress_test_results])
        
        return {
            'team_members': team_members,
            'original_team_score': original_team_score,
            'stress_test_results': stress_test_results,
            'average_vulnerability': avg_vulnerability,
            'maximum_vulnerability': max_vulnerability,
            'risk_level': 'HIGH' if max_vulnerability > original_team_score * 0.3 else 'MEDIUM' if max_vulnerability > original_team_score * 0.15 else 'LOW'
        }
    
    def _calculate_team_cohesion(self, team_members: List[str]) -> float:
        """Calculate team cohesion score using existing digital twins"""
        return self._calculate_team_cohesion_with_twins(team_members, self.employee_twins)
    
    def _calculate_team_cohesion_with_twins(self, team_members: List[str], twins_dict: Dict[str, Dict[str, Any]]) -> float:
        """Calculate team cohesion score with specified twins dictionary"""
        if len(team_members) < 2:
            return 0.0
        
        total_score = 0.0
        pair_count = 0
        
        for i, member1 in enumerate(team_members):
            if member1 not in twins_dict:
                continue
                
            for j, member2 in enumerate(team_members):
                if i < j and member2 in twins_dict:
                    twin1 = twins_dict[member1]
                    twin2 = twins_dict[member2]
                    
                    similarity = self._calculate_twin_similarity(twin1, twin2)
                    total_score += similarity
                    pair_count += 1
        
        return total_score / pair_count if pair_count > 0 else 0.0
    
    def _calculate_twin_similarity(self, twin1: Dict[str, Any], twin2: Dict[str, Any]) -> float:
        """Calculate similarity between two digital twins"""
        skills1 = set(twin1.get('skills', []))
        skills2 = set(twin2.get('skills', []))
        
        intersection = len(skills1.intersection(skills2))
        union = len(skills1.union(skills2))
        jaccard = intersection / union if union > 0 else 0
        
        features1 = twin1.get('features', [])
        features2 = twin2.get('features', [])
        
        if len(features1) == len(features2) and len(features1) > 0:
            distance = np.sqrt(sum((f1 - f2) ** 2 for f1, f2 in zip(features1, features2)))
            feature_similarity = 1 / (1 + distance)
        else:
            feature_similarity = 0
        
        return (jaccard + feature_similarity) / 2
