"""
Benchmark script to measure performance of COMPASS components
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import random
from compass.digital_twin import DigitalTwinEngine
from compass.scenario_simulator import ScenarioSimulator


def generate_sample_employee_data(num_employees: int = 100) -> List[Dict[str, Any]]:
    """Generate sample employee data for testing"""
    employees = []
    
    skills_pool = ['python', 'javascript', 'sql', 'machine_learning', 'data_analysis', 
                   'project_management', 'communication', 'leadership', 'design', 'marketing']
    
    for i in range(num_employees):
        employee = {
            'employee_id': f'emp_{i:03d}',
            'skills': random.sample(skills_pool, random.randint(2, 6)),
            'performance': {
                'quarterly_reviews': [random.uniform(3.0, 5.0) for _ in range(4)],
                'project_completions': random.randint(5, 20),
                'peer_ratings': [random.uniform(3.5, 5.0) for _ in range(3)]
            },
            'collaboration_data': {
                'email_interactions': [random.randint(10, 100) for _ in range(12)],
                'meeting_participation': [random.uniform(0.5, 1.0) for _ in range(12)],
                'cross_team_projects': random.randint(1, 8)
            },
            'demographics': {
                'tenure_years': random.uniform(0.5, 15.0),
                'department': random.choice(['engineering', 'marketing', 'sales', 'hr', 'finance']),
                'level': random.choice(['junior', 'mid', 'senior', 'lead', 'manager'])
            }
        }
        employees.append(employee)
    
    return employees


def benchmark_digital_twin_creation(num_employees: int = 100) -> Dict[str, float]:
    """Benchmark digital twin creation performance"""
    print(f"Benchmarking digital twin creation with {num_employees} employees...")
    
    engine = DigitalTwinEngine()
    employees_data = generate_sample_employee_data(num_employees)
    
    start_time = time.time()
    twins = engine.batch_create_twins(employees_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    time_per_employee = execution_time / num_employees
    
    print(f"Total time: {execution_time:.3f} seconds")
    print(f"Time per employee: {time_per_employee:.6f} seconds")
    
    return {
        'total_time': execution_time,
        'time_per_employee': time_per_employee,
        'num_employees': num_employees
    }


def benchmark_scenario_simulation(num_employees: int = 50, num_simulations: int = 100) -> Dict[str, float]:
    """Benchmark scenario simulation performance"""
    print(f"Benchmarking scenario simulation with {num_employees} employees, {num_simulations} simulations...")
    
    engine = DigitalTwinEngine()
    employees_data = generate_sample_employee_data(num_employees)
    twins = engine.batch_create_twins(employees_data)
    
    simulator = ScenarioSimulator(twins)
    
    start_time = time.time()
    results = simulator.simulate_layoff_scenario(layoff_percentage=0.2, num_simulations=num_simulations)
    end_time = time.time()
    
    execution_time = end_time - start_time
    time_per_simulation = execution_time / num_simulations
    
    print(f"Total time: {execution_time:.3f} seconds")
    print(f"Time per simulation: {time_per_simulation:.6f} seconds")
    
    return {
        'total_time': execution_time,
        'time_per_simulation': time_per_simulation,
        'num_simulations': num_simulations,
        'num_employees': num_employees
    }


def benchmark_monte_carlo_optimization(num_employees: int = 30, num_iterations: int = 1000) -> Dict[str, float]:
    """Benchmark Monte Carlo optimization performance"""
    print(f"Benchmarking Monte Carlo optimization with {num_employees} employees, {num_iterations} iterations...")
    
    engine = DigitalTwinEngine()
    employees_data = generate_sample_employee_data(num_employees)
    twins = engine.batch_create_twins(employees_data)
    
    simulator = ScenarioSimulator(twins)
    constraints = {
        'max_team_size': 8,
        'min_skill_diversity': 0.3
    }
    
    start_time = time.time()
    results = simulator.monte_carlo_optimization(constraints, num_iterations)
    end_time = time.time()
    
    execution_time = end_time - start_time
    time_per_iteration = execution_time / num_iterations
    
    print(f"Total time: {execution_time:.3f} seconds")
    print(f"Time per iteration: {time_per_iteration:.6f} seconds")
    
    return {
        'total_time': execution_time,
        'time_per_iteration': time_per_iteration,
        'num_iterations': num_iterations,
        'num_employees': num_employees
    }


def run_all_benchmarks():
    """Run comprehensive benchmarks"""
    print("=" * 60)
    print("COMPASS Performance Benchmarks")
    print("=" * 60)
    
    benchmarks = {}
    
    benchmarks['digital_twin_small'] = benchmark_digital_twin_creation(50)
    benchmarks['digital_twin_medium'] = benchmark_digital_twin_creation(100)
    benchmarks['digital_twin_large'] = benchmark_digital_twin_creation(200)
    
    print("\n" + "-" * 40)
    
    benchmarks['simulation_small'] = benchmark_scenario_simulation(25, 50)
    benchmarks['simulation_medium'] = benchmark_scenario_simulation(50, 100)
    
    print("\n" + "-" * 40)
    
    benchmarks['optimization_small'] = benchmark_monte_carlo_optimization(20, 500)
    benchmarks['optimization_medium'] = benchmark_monte_carlo_optimization(30, 1000)
    
    print("\n" + "=" * 60)
    print("Benchmark Summary:")
    print("=" * 60)
    
    for name, results in benchmarks.items():
        print(f"{name}: {results}")
    
    return benchmarks


if __name__ == "__main__":
    run_all_benchmarks()
