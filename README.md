# EpiSim-Urbanza-zhazha-
EpiSim-Urbanza-zhazha  is a comprehensive epidemiological modeling framework that simulates the impact of public health interventions on urban disease outbreaks. 

Core Innovation: The Spatiotemporal-Behavioral Hybrid Model represents a major breakthrough, evolving from traditional uniform SIR models to dynamically adaptive models accounting for spatial heterogeneity.

## 🚀 Quick Start Guide

### Environment Setup
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Verify Installation
python -c "import numpy, pandas, matplotlib, seaborn; print('✅ Environment setup completed')"
```

### Run the Project
```bash
# Main Demo (Recommended for First Run)
python demo.py

# Generate All Visualizations
python run_all_visualizations.py

# Run Functional Tests
python test_all_functions.py
```

## 💡 Core Model Explanation

### 1. Basic SIR Model (`src/models/sir_model.py`)

**Function:** Implements a time-discrete SIR epidemiological dynamics model.

**Core Code Example:**
```python
from src.models.sir_model import SIRModel

# Create the model
model = SIRModel(
    N=10000,        # Total population
    beta=0.30,      # Transmission rate (R0 = β/γ = 3.0)
    gamma=0.10,     # Recovery rate (average infectious period = 10 days)
    I0=10,          # Initial infected individuals
    R0=0            # Initial recovered individuals
)

# Run the simulation
results = model.simulate(days=120)
print(f"Peak infection proportion: {results['I_prop'].max():.3f}")
```

**Output Variables**:
- `S_prop`: Proportion of Susceptible Individuals Time Series
- `I_prop`: Infection Rate Time Series  
- `R_prop`: Recovery Rate Time Series
- `day`: Time steps corresponding to each iteration

### 2. SEIR Extended Model (`src/models/seir_model.py`)

**Function:**  
Implements an SEIR model that adds an *exposed (latent)* compartment to the basic SIR framework.

**Core Code Example:**
```python
from src.models.seir_model import SEIRModel

# Create the SEIR model
model = SEIRModel(
    N=10000,        # Total population
    beta=0.30,      # Transmission rate
    sigma=0.20,     # Incubation rate (average incubation period = 5 days)
    gamma=0.10,     # Recovery rate
    I0=10,          # Initial infected individuals
    E0=0,           # Initial exposed individuals
    R0=0            # Initial recovered individuals
)

results = model.simulate(120)
```
Key Difference:
The SEIR model generally produces a 35–40% lower peak infection rate compared to the SIR model,
and the peak occurs later due to the inclusion of the latent (exposed) period.

### 3. 🔥 Hybrid Spatial-Behavior Model (`src/models/hybrid_model.py`)

**Core Innovation**: The project's standout feature, integrating spatial heterogeneity and dynamic behavioral adaptation

**Functional Features**:
- **4 Spatial Zones**: Urban core, suburbs, rural areas, industrial zones
- **Dynamic Behavioral Adaptation**: Risk-perception-based adjustment of contact rates
- **Inter-regional Mobility**: Population movement between different zones
- **Threshold-Triggered Interventions**: Automatic activation of behavior modification measures

**Core Code**:
```python
from src.models.hybrid_model import HybridEpidemicModel

# Create a hybrid model
model = HybridEpidemicModel(
    N=10000,
    n_zones=4,                    # 4 spatial zones
    base_beta=0.30,
    base_gamma=0.10,
    adaptation_factor=0.5,        # Behavioral Adaptation Intensity
    risk_threshold=0.05,          # Risk Perception Threshold
    I0=10,
    R0=0
)

# Run Simulation
results = model.simulate(120)

# Acquire Innovation Metrics
innovation_metrics = model.get_innovation_metrics(results)
print(f"Behavioral Adaptation Kickoff Day: {innovation_metrics['behavioral_adaptation_day']}")
print(f"Regional Peak Heterogeneity: {innovation_metrics['peak_heterogeneity']:.3f}")
```
**Output**:
```python
{
    'aggregate_results': DataFrame,      # Overall results
    'zone_results': [DataFrame],         # Results by Region
    'adaptation_history': DataFrame,     # Behavioral Adaptation to History
    'zone_populations': [int],          # Population by Region
    'final_sizes': [float]              # Final scale of the epidemic in each region
}
```

### 4. Network Propagation Model (`src/models/network_model.py`)

**Functionality**: Agent simulation based on complex networks

**Supported Network Types**:
- **Small-world networks**: High clustering coefficient + Short path length
- **Scale-free networks**: Power-law degree distribution
- **Random Networks**: Poisson degree distribution

**Core Code**:
```python
from src.models.network_model import NetworkEpidemicModel

# Create a network model
model = NetworkEpidemicModel(
    N=1000,                    # Network scale
    network_type='small_world', # Network Type
    infection_prob=0.05,       # Probability of infection
    recovery_prob=0.10,        # Probability of recovery
    I0=5
)

results = model.simulate(100)
network_props = model.get_network_properties()
```

## 🎯 Intervention Scenario System

### Scenario Types (`src/scenarios/`)

**1. Baseline Scenario**:
```python
from src.scenarios.baseline import BaselineScenario

scenario = BaselineScenario()
results = scenario.run_simulation(120)
metrics = scenario.get_metrics(results)
```

**2. Reduced Contact Scenarios**:
```python
from src.scenarios.interventions import ContactReductionScenario

# 30% reduction in contact
scenario = ContactReductionScenario(reduction_percent=0.30)
results = scenario.run_simulation(120)
# Effect: 44% reduction in infection peaks
```

**3. Rapid Isolation Scenarios**:
```python
from src.scenarios.interventions import FasterIsolationScenario

# Recovery rate increased by 50%
scenario = FasterIsolationScenario(gamma_multiplier=1.5)
results = scenario.run_simulation(120)
# Effect: 35% reduction in infection peaks
```

**4. Vaccination Scenarios**:
```python
from src.scenarios.interventions import VaccinationScenario

# 40% of the population pre-immunized
scenario = VaccinationScenario(vaccination_rate=0.40)
results = scenario.run_simulation(120)
# Effect: 77% reduction in infection peaks
```

**5. Combined Intervention Scenario**:
```python
from src.scenarios.interventions import CombinedInterventionScenario

scenario = CombinedInterventionScenario()
results = scenario.run_simulation(120)
# Effect: 99.3% reduction in infection peaks
```

## 📊 Analysis and Indicator System

### Core Metric Calculation (`src/analysis/metrics.py`)

```python
from src.analysis.metrics import calculate_all_metrics, compare_scenarios

# Single-scenario metric
metrics = calculate_all_metrics(results, N=10000)

# Key Indicators
print(f"Peak infection rate: {metrics['peak_infection_proportion']:.3f}")
print(f"peak hours: {metrics['time_to_peak']:.0f} day")
print(f"Ultimate scale of the epidemic: {metrics['final_epidemic_size']:.3f}")
print(f"Duration of the pandemic: {metrics['epidemic_duration']:.0f} day")

# Multi-scenario Comparison
scenario_results = {
    'Baseline': baseline_results,
    'Reduced contact': contact_reduction_results,
    'Vaccination': vaccination_results
}

comparison_df = compare_scenarios(scenario_results, N=10000)
print(comparison_df)
```

### Sensitivity Analysis (`src/analysis/sensitivity.py`)

```python
from src.analysis.sensitivity import SensitivityAnalyzer

# Create Analyzer
analyzer = SensitivityAnalyzer(base_params={
    'N': 10000, 'beta': 0.30, 'gamma': 0.10, 'I0': 10, 'R0': 0
})

# β Parameter Sensitivity
beta_range = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
beta_analysis = analyzer.analyze_parameter('beta', beta_range)

# Comprehensive Sensitivity Analysis
comprehensive = analyzer.comprehensive_sensitivity_analysis()
```

## 🎨 Visualization System

### Basic Drawing (`utils/visualization.py`)

```python
from utils.visualization import (
    plot_sir_timeseries, plot_seir_timeseries,
    plot_infection_comparison, plot_metrics_comparison
)

# SIR Time Series Chart
fig = plot_sir_timeseries(sir_results, "SIR Model Dynamics")
plt.show()

# Scenario Comparison Chart
fig = plot_infection_comparison(scenario_results, "Comparison of Intervention Effects")
plt.show()

# Indicator Comparison Chart
fig = plot_metrics_comparison(comparison_df, "Key Metrics Comparison")
plt.show()
```

## 🔧 Manual Run Example

### Complete Workflow

```python
#!/usr/bin/env python3
"""Complete Manual Operation Example"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.sir_model import SIRModel
from src.models.hybrid_model import HybridEpidemicModel
from src.scenarios.interventions import ContactReductionScenario
from src.analysis.metrics import compare_scenarios
from utils.visualization import plot_infection_comparison

def complete_analysis():
    """Complete Analysis Process"""
    
    # 1. Foundational Model Comparison
    print("=== Foundation Model Comparison ===")
    sir_model = SIRModel(N=10000, beta=0.30, gamma=0.10, I0=10)
    sir_results = sir_model.simulate(120)
    
    hybrid_model = HybridEpidemicModel(N=10000, base_beta=0.30, base_gamma=0.10, I0=10)
    hybrid_results = hybrid_model.simulate(120)
    
    print(f"Standard SIR Peak: {sir_results['I_prop'].max():.3f}")
    print(f"Peak of the Mixed Model: {hybrid_results['aggregate_results']['I_prop'].max():.3f}")
    
    # 2. Intervention Scenario Analysis
    print("\n=== Intervention Scenario Analysis ===")
    scenarios = {
        'Baseline': sir_results,
        'Hybrid model': hybrid_results['aggregate_results'],
        'Reduction in contact by 30%': ContactReductionScenario(0.30).run_simulation(120)
    }
    
    # 3. Generate Comparative Analysis
    comparison_df = compare_scenarios(scenarios, 10000)
    print(comparison_df)
    
    # 4. Visualization
    fig = plot_infection_comparison(scenarios, "Comprehensive Comparative Analysis")
    plt.savefig('complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scenarios, comparison_df

if __name__ == "__main__":
    scenarios, comparison = complete_analysis()
```

## 📱 Using Jupyter Notebook

### Primary Analysis Notebook (`notebooks/main_analysis.ipynb`)

```python
# Run in Jupyter
import sys
sys.path.append('..')

# Import all modules
from src.models import *
from src.scenarios import *
from src.analysis import *
from utils.visualization import *

# Interactive Analysis
%matplotlib inline
plt.style.use('seaborn-v0_8')

# Run a complete analysis
# ... (Detailed Analysis of the Code)
```

### Quick Test Notebook (`notebooks/quick_test.ipynb`)

For rapid functional verification and parameter debugging.

## 📁 Output File Specifications

### Files generated after execution

**Data file**:
- `data/demo_results.csv` - Demonstrate the results
- `data/scenario_comparison.png` - Scenario Comparison Chart
- `data/sir_sensitivity_summary.csv` - SIR Sensitivity Analysis
- `data/seir_sensitivity_summary.csv` - SEIR Sensitivity Analysis

**Visualization File** (After running `run_all_visualizations.py`):
- `all_visualizations/01_basic_models.png` - Foundation Model Comparison
- `all_visualizations/02_intervention_scenarios.png` - Intervention Scenario
- `all_visualizations/03_intervention_metrics.png` - Effectiveness Metrics
- `all_visualizations/04_hybrid_innovation.png` - Innovation in Mixed Models
- `all_visualizations/05_sensitivity_analysis.png` - Sensitivity Analysis
- `all_visualizations/06_network_models.png` - Comparison of Network Models
- `all_visualizations/07_summary_dashboard.png` - Comprehensive Dashboard

## 🎯 Key Findings and Results

### Primary Study Findings

1. **Contact Reduction Effect**: 30% reduction in contacts → 44% peak reduction
2. **Rapid Isolation Effect**: 50% increase in recovery rate → 35% peak reduction
3. **Vaccination Effect**: 40% pre-immunization → 77% peak reduction
4. **Combined Intervention Effect**: Multiple interventions → 99.3% peak reduction
5. **Hybrid Model Innovation**: Spatial heterogeneity + behavioral adaptation → 49% additional peak reduction

### Advantages of the Innovative Model

- **Realism**: Incorporates spatial heterogeneity and population behavior
- **Dynamism**: Enables real-time risk perception and adaptation
- **Policy Relevance**: Provides tailored intervention recommendations
- **Scalability**: Supports diverse network topologies and parameter configurations

## 🔍 Troubleshooting

### Common Error

**1. Import error**:
```bash
# Ensure the command is run in the project's root directory.
cd /path/to/project
python demo.py
```

**2. Visualization issues**:
```python
# Set the matplotlib backend
import matplotlib
matplotlib.use('Agg')  # Server Environment
```

**3. Insufficient memory**:
```python
# Reduce the scale of simulation
model = SIRModel(N=1000)  # Not 10,000
```

## 📚 Extended Use

### Custom Scenes

```python
class CustomScenario:
    def __init__(self, custom_param):
        self.custom_param = custom_param
    
    def run_simulation(self, days=120):
        # Custom Analog Logic
        model = SIRModel(beta=self.custom_param)
        return model.simulate(days)
    
    def get_metrics(self, results):
        # Custom Metric Calculation
        return calculate_all_metrics(results, 10000)
```

### Parameter Optimization

```python
from scipy.optimize import minimize

def objective_function(params):
    """Optimization objective function"""
    beta, gamma = params
    model = SIRModel(N=10000, beta=beta, gamma=gamma, I0=10)
    results = model.simulate(120)
    return results['I_prop'].max()  # Minimize Peak Values

# Operational Optimization
result = minimize(objective_function, [0.3, 0.1], 
                 bounds=[(0.1, 0.5), (0.05, 0.2)])
```

## 🏆 Project Summary

This epidemiological modeling project achieves a major breakthrough from traditional SIR/SEIR models to realistic complex system modeling through an innovative hybrid spatial-behavioral model. The project not only provides a comprehensive modeling toolchain but also offers scientific basis and quantitative support for pandemic control policy formulation.

**Core Values**:
- 🔬 **Scientific Value**: Methodological innovation and theoretical contributions
- 🏛️ **Policy Value**: Supporting public health decision-making
- 🛠️ **Engineering Value**: A comprehensive open-source modeling framework
- 📚 **Educational Value**: Exceptional teaching and learning resources

---

*Last Updated: October 2025*
*Author: [Group32]*
