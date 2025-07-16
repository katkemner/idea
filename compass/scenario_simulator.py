"""
Scenario Simulator - Monte Carlo + agent-based modeling for organizational changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import random
from itertools import combinations
import networkx as nx
from deap import base, creator, tools, algorithms


class ScenarioSimulator:
    """
    Simulates strategic HR scenarios using Monte Carlo and agent-based modeling.
    
    Contains several efficiency issues for demonstration and improvement.
    """
    
    def __init__(self, digital_twins: Dict[str, Dict[str, Any]]):
        self.digital_twins = digital_twins
        self.simulation_results = []
        self.collaboration_graph = None
        self._setup_nsga2()
        
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
    
    def _setup_nsga2(self):
        """Setup NSGA-II genetic algorithm components"""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selNSGA2)
    
    def nsga2_optimization(self, constraints: Dict[str, Any], 
                          population_size: int = 50, generations: int = 100) -> Dict[str, Any]:
        """
        NSGA-II Multi-Objective Optimization for team composition.
        
        Replaces Monte Carlo with sophisticated genetic algorithms that optimize
        multiple objectives simultaneously: productivity, diversity, collaboration.
        """
        self.constraints = constraints
        employee_ids = list(self.digital_twins.keys())
        
        if len(employee_ids) < 4:
            return self.monte_carlo_optimization(constraints, 1000)
        
        population = self.toolbox.population(n=population_size)
        
        for individual in population:
            individual.fitness.values = self.toolbox.evaluate(individual)
        
        for generation in range(generations):
            offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.7, mutpb=0.3)
            
            for child in offspring:
                child.fitness.values = self.toolbox.evaluate(child)
            
            population = self.toolbox.select(population + offspring, population_size)
        
        best_individual = tools.selBest(population, 1)[0]
        best_config = self._individual_to_config(best_individual)
        best_score = sum(best_individual.fitness.values)
        
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        
        return {
            'best_configuration': best_config,
            'best_score': best_score,
            'pareto_front_size': len(pareto_front),
            'optimization_method': 'NSGA-II',
            'generations': generations,
            'population_size': population_size,
            'objectives': {
                'productivity': best_individual.fitness.values[0],
                'diversity': best_individual.fitness.values[1], 
                'collaboration': best_individual.fitness.values[2]
            }
        }
    
    def _create_individual(self):
        """Create a random individual (team configuration) for NSGA-II"""
        employee_ids = list(self.digital_twins.keys())
        individual = creator.Individual()
        
        remaining_employees = employee_ids.copy()
        num_teams = random.randint(2, min(8, len(employee_ids) // 2))
        
        for team_idx in range(num_teams):
            if not remaining_employees:
                break
                
            max_team_size = self.constraints.get('max_team_size', 8)
            team_size = random.randint(1, min(max_team_size, len(remaining_employees)))
            
            team_members = random.sample(remaining_employees, team_size)
            for member in team_members:
                remaining_employees.remove(member)
            
            individual.append(team_members)
        
        return individual
    
    def _individual_to_config(self, individual: List[List[str]]) -> Dict[str, List[str]]:
        """Convert NSGA-II individual to team configuration format"""
        config = {}
        for i, team_members in enumerate(individual):
            config[f'team_{i}'] = team_members
        return config
    
    def _evaluate_individual(self, individual: List[List[str]]) -> Tuple[float, float, float]:
        """
        Multi-objective fitness evaluation for NSGA-II.
        
        Returns tuple of (productivity, diversity, collaboration) scores.
        """
        productivity_score = 0.0
        diversity_score = 0.0
        collaboration_score = 0.0
        
        for team_members in individual:
            if not team_members:
                continue
            
            team_productivity = 0.0
            team_skills = []
            team_collaboration = 0.0
            
            for member_id in team_members:
                if member_id in self.digital_twins:
                    twin = self.digital_twins[member_id]
                    perf_metrics = twin.get('performance_metrics', {})
                    team_productivity += perf_metrics.get('mean', 0)
                    team_skills.extend(twin.get('skills', []))
            
            skill_diversity = len(set(team_skills)) / len(team_skills) if team_skills else 0
            
            for i, member1 in enumerate(team_members):
                for j, member2 in enumerate(team_members):
                    if i < j and member1 in self.digital_twins and member2 in self.digital_twins:
                        similarity = self._calculate_similarity(
                            self.digital_twins[member1],
                            self.digital_twins[member2]
                        )
                        team_collaboration += similarity
            
            productivity_score += team_productivity
            diversity_score += skill_diversity
            collaboration_score += team_collaboration
        
        return (productivity_score, diversity_score, collaboration_score)
    
    def _crossover(self, ind1: List[List[str]], ind2: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """Crossover operation for NSGA-II team configurations"""
        if len(ind1) == 0 or len(ind2) == 0:
            return ind1, ind2
        
        point1 = random.randint(0, len(ind1) - 1)
        point2 = random.randint(0, len(ind2) - 1)
        
        new_ind1 = creator.Individual(ind1[:point1] + ind2[point2:])
        new_ind2 = creator.Individual(ind2[:point2] + ind1[point1:])
        
        return new_ind1, new_ind2
    
    def _mutate(self, individual: List[List[str]]) -> Tuple[List[List[str]]]:
        """Mutation operation for NSGA-II team configurations"""
        if not individual or random.random() > 0.1:
            return (individual,)
        
        team_idx = random.randint(0, len(individual) - 1)
        team = individual[team_idx]
        
        if len(team) > 1:
            member_to_remove = random.choice(team)
            team.remove(member_to_remove)
            
            other_team_idx = random.randint(0, len(individual) - 1)
            if other_team_idx != team_idx:
                individual[other_team_idx].append(member_to_remove)
        
        return (individual,)
    
    def build_collaboration_graph(self) -> nx.Graph:
        """
        Build weighted collaboration graph from digital twin data.
        
        Uses NetworkX to model employee relationships based on collaboration metrics,
        enabling graph-based team formation and network analysis.
        """
        G = nx.Graph()
        
        employee_ids = list(self.digital_twins.keys())
        
        for emp_id in employee_ids:
            twin = self.digital_twins[emp_id]
            G.add_node(emp_id, 
                      skills=twin.get('skills', []),
                      collaboration_score=twin.get('collaboration_score', 0),
                      performance=twin.get('performance_metrics', {}).get('mean', 0))
        
        for i, emp1 in enumerate(employee_ids):
            for j, emp2 in enumerate(employee_ids):
                if i < j:
                    twin1 = self.digital_twins[emp1]
                    twin2 = self.digital_twins[emp2]
                    
                    similarity = self._calculate_similarity(twin1, twin2)
                    
                    if similarity > 0.1:
                        G.add_edge(emp1, emp2, weight=similarity)
        
        self.collaboration_graph = G
        return G
    
    def analyze_collaboration_network(self) -> Dict[str, Any]:
        """
        Analyze collaboration network using graph metrics.
        
        Provides network centrality measures for enhanced team formation decisions.
        """
        if self.collaboration_graph is None:
            self.build_collaboration_graph()
        
        G = self.collaboration_graph
        
        centrality_measures = {
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'degree': nx.degree_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000)
        }
        
        clustering_coefficient = nx.clustering(G)
        
        connected_components = list(nx.connected_components(G))
        
        key_connectors = sorted(centrality_measures['betweenness'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        influential_nodes = sorted(centrality_measures['eigenvector'].items(),
                                  key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'network_stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'connected_components': len(connected_components)
            },
            'centrality_measures': centrality_measures,
            'clustering_coefficient': clustering_coefficient,
            'key_connectors': key_connectors,
            'influential_nodes': influential_nodes,
            'average_clustering': nx.average_clustering(G)
        }
    
    def optimize_teams_with_graph_analysis(self, constraints: Dict[str, Any], 
                                         use_centrality: bool = True) -> Dict[str, Any]:
        """
        Enhanced team optimization using graph-based relationship modeling.
        
        Combines NSGA-II optimization with network centrality measures for
        superior team formation decisions.
        """
        if self.collaboration_graph is None:
            self.build_collaboration_graph()
        
        network_analysis = self.analyze_collaboration_network()
        
        if use_centrality:
            centrality_weights = network_analysis['centrality_measures']['betweenness']
            self._apply_centrality_weights(centrality_weights)
        
        optimization_result = self.nsga2_optimization(constraints)
        
        optimization_result['network_analysis'] = network_analysis
        optimization_result['used_graph_centrality'] = use_centrality
        
        return optimization_result
    
    def _apply_centrality_weights(self, centrality_weights: Dict[str, float]):
        """Apply network centrality weights to digital twin performance metrics"""
        for emp_id, centrality in centrality_weights.items():
            if emp_id in self.digital_twins:
                current_performance = self.digital_twins[emp_id]['performance_metrics'].get('mean', 0)
                boosted_performance = current_performance * (1 + centrality * 0.5)
                self.digital_twins[emp_id]['performance_metrics']['mean'] = boosted_performance
