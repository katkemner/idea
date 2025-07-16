"""
Scenario Simulator - Monte Carlo + agent-based modeling for organizational changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import random
from itertools import combinations


class ScenarioSimulator:
    """
    Simulates strategic HR scenarios using Monte Carlo and agent-based modeling.
    
    Contains several efficiency issues for demonstration and improvement.
    """
    
    def __init__(self, digital_twins: Dict[str, Dict[str, Any]]):
        self.digital_twins = digital_twins
        self.simulation_results = []
        
    def simulate_layoff_scenario(self, layoff_percentage: float, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Simulate layoff scenarios.
        
        EFFICIENCY ISSUE #7: Inefficient random sampling and simulation
        """
        results = []
        employee_ids = list(self.digital_twins.keys())
        
        for sim in range(num_simulations):
            num_layoffs = int(len(employee_ids) * layoff_percentage)
            
            laid_off = []
            remaining_employees = employee_ids.copy()
            for _ in range(num_layoffs):
                if remaining_employees:
                    chosen = random.choice(remaining_employees)
                    laid_off.append(chosen)
                    remaining_employees.remove(chosen)
            
            remaining_skills = []
            total_productivity = 0
            collaboration_impact = 0
            
            for emp_id in remaining_employees:
                twin = self.digital_twins[emp_id]
                remaining_skills.extend(twin.get('skills', []))
                
                for other_emp_id in remaining_employees:
                    if emp_id != other_emp_id:
                        other_twin = self.digital_twins[other_emp_id]
                        similarity = self._calculate_similarity(twin, other_twin)
                        collaboration_impact += similarity
                
                perf_metrics = twin.get('performance_metrics', {})
                total_productivity += perf_metrics.get('mean', 0)
            
            unique_skills = len(set(remaining_skills))
            skills_lost = len(set([skill for emp_id in laid_off 
                                 for skill in self.digital_twins[emp_id].get('skills', [])]))
            
            results.append({
                'simulation_id': sim,
                'laid_off_count': len(laid_off),
                'remaining_count': len(remaining_employees),
                'skills_retained': unique_skills,
                'skills_lost': skills_lost,
                'total_productivity': total_productivity,
                'collaboration_score': collaboration_impact
            })
        
        return self._aggregate_simulation_results(results)
    
    def _calculate_similarity(self, twin1: Dict[str, Any], twin2: Dict[str, Any]) -> float:
        """
        EFFICIENCY ISSUE #8: Inefficient similarity calculation
        """
        skills1 = set(twin1.get('skills', []))
        skills2 = set(twin2.get('skills', []))
        
        intersection = len(skills1.intersection(skills2))
        union = len(skills1.union(skills2))
        
        jaccard = intersection / union if union > 0 else 0
        
        features1 = twin1.get('features', [])
        features2 = twin2.get('features', [])
        
        if len(features1) == len(features2) and len(features1) > 0:
            distance = 0
            for i in range(len(features1)):
                distance += (features1[i] - features2[i]) ** 2
            feature_similarity = 1 / (1 + np.sqrt(distance))
        else:
            feature_similarity = 0
        
        return (jaccard + feature_similarity) / 2
    
    def simulate_reorganization(self, new_team_structure: Dict[str, List[str]], 
                              num_simulations: int = 500) -> Dict[str, Any]:
        """
        Simulate organizational reorganization.
        
        EFFICIENCY ISSUE #9: Inefficient team optimization
        """
        results = []
        
        for sim in range(num_simulations):
            team_scores = {}
            
            for team_name, member_ids in new_team_structure.items():
                team_score = 0
                team_skills = []
                
                for member_id in member_ids:
                    if member_id in self.digital_twins:
                        twin = self.digital_twins[member_id]
                        team_skills.extend(twin.get('skills', []))
                        
                        perf_metrics = twin.get('performance_metrics', {})
                        team_score += perf_metrics.get('mean', 0)
                
                for i, member1 in enumerate(member_ids):
                    for j, member2 in enumerate(member_ids):
                        if i < j and member1 in self.digital_twins and member2 in self.digital_twins:
                            similarity = self._calculate_similarity(
                                self.digital_twins[member1], 
                                self.digital_twins[member2]
                            )
                            team_score += similarity * 0.1
                
                unique_skills = len(set(team_skills))
                skill_diversity = unique_skills / len(team_skills) if team_skills else 0
                
                team_scores[team_name] = {
                    'performance_score': team_score,
                    'skill_diversity': skill_diversity,
                    'team_size': len(member_ids)
                }
            
            results.append({
                'simulation_id': sim,
                'team_scores': team_scores,
                'overall_score': sum(scores['performance_score'] for scores in team_scores.values())
            })
        
        return self._aggregate_simulation_results(results)
    
    def _aggregate_simulation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        EFFICIENCY ISSUE #10: Inefficient result aggregation
        """
        if not results:
            return {}
        
        aggregated = {}
        
        for key in results[0].keys():
            if key != 'simulation_id' and isinstance(results[0][key], (int, float)):
                values = []
                for result in results:
                    values.append(result[key])
                
                aggregated[f'{key}_mean'] = sum(values) / len(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = min(values)
                aggregated[f'{key}_max'] = max(values)
                
                sorted_values = sorted(values)
                p25_idx = int(0.25 * len(sorted_values))
                p75_idx = int(0.75 * len(sorted_values))
                aggregated[f'{key}_p25'] = sorted_values[p25_idx]
                aggregated[f'{key}_p75'] = sorted_values[p75_idx]
        
        aggregated['num_simulations'] = len(results)
        return aggregated
    
    def monte_carlo_optimization(self, constraints: Dict[str, Any], 
                               num_iterations: int = 10000) -> Dict[str, Any]:
        """
        EFFICIENCY ISSUE #11: Inefficient Monte Carlo optimization
        """
        best_score = float('-inf')
        best_configuration = None
        employee_ids = list(self.digital_twins.keys())
        
        for iteration in range(num_iterations):
            config = {}
            remaining_employees = employee_ids.copy()
            
            num_teams = random.randint(2, min(10, len(employee_ids) // 2))
            for team_idx in range(num_teams):
                team_size = random.randint(1, len(remaining_employees) // (num_teams - team_idx))
                team_members = []
                
                for _ in range(min(team_size, len(remaining_employees))):
                    if remaining_employees:
                        member = random.choice(remaining_employees)
                        team_members.append(member)
                        remaining_employees.remove(member)
                
                config[f'team_{team_idx}'] = team_members
            
            score = self._evaluate_configuration(config, constraints)
            
            if score > best_score:
                best_score = score
                best_configuration = config.copy()
        
        return {
            'best_configuration': best_configuration,
            'best_score': best_score,
            'iterations': num_iterations
        }
    
    def _evaluate_configuration(self, config: Dict[str, List[str]], 
                              constraints: Dict[str, Any]) -> float:
        """
        EFFICIENCY ISSUE #12: Inefficient configuration evaluation
        """
        total_score = 0
        
        for team_name, members in config.items():
            if not members:
                continue
                
            team_productivity = 0
            team_skills = []
            
            for member_id in members:
                if member_id in self.digital_twins:
                    twin = self.digital_twins[member_id]
                    perf_metrics = twin.get('performance_metrics', {})
                    team_productivity += perf_metrics.get('mean', 0)
                    team_skills.extend(twin.get('skills', []))
            
            skill_diversity = len(set(team_skills)) / len(team_skills) if team_skills else 0
            
            collaboration_score = 0
            for i, member1 in enumerate(members):
                for j, member2 in enumerate(members):
                    if i < j:
                        if member1 in self.digital_twins and member2 in self.digital_twins:
                            similarity = self._calculate_similarity(
                                self.digital_twins[member1],
                                self.digital_twins[member2]
                            )
                            collaboration_score += similarity
            
            team_score = team_productivity * 0.4 + skill_diversity * 0.3 + collaboration_score * 0.3
            
            if 'max_team_size' in constraints:
                if len(members) > constraints['max_team_size']:
                    team_score *= 0.5
            
            if 'min_skill_diversity' in constraints:
                if skill_diversity < constraints['min_skill_diversity']:
                    team_score *= 0.7
            
            total_score += team_score
        
        return total_score
