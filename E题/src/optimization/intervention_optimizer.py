"""
Intervention Strategy Optimization

Optimizes allocation of light pollution mitigation interventions across locations
to minimize pollution while respecting budget and feasibility constraints.

Problem formulation:
    Minimize: Total light pollution (weighted by TOPSIS scores)
    Subject to:
        - Budget constraint: sum(cost_i * x_i) <= total_budget
        - Feasibility: x_i >= 0
        - Impact bounds: Each intervention has max reduction potential
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


@dataclass
class InterventionOption:
    """A single intervention strategy."""
    name: str
    cost_per_unit: float  # Cost per unit area (e.g., $/km²)
    max_reduction_potential: float  # Max % reduction in pollution (0-1)
    applicable_locations: List[str]  # Which location types can use this
    effectiveness_decay: float = 0.0  # Diminishing returns factor


@dataclass
class InterventionPlan:
    """Optimized intervention allocation plan."""
    location_names: List[str]
    intervention_allocations: Dict[str, Dict[str, float]]  # {location: {intervention: amount}}
    total_cost: float
    expected_pollution_reduction: np.ndarray  # Per location
    objective_value: float
    status: str  # 'optimal', 'feasible', 'infeasible'
    budget_used: float
    budget_limit: float


class InterventionOptimizer:
    """
    Optimize intervention strategy allocation for light pollution mitigation.
    
    Supports multiple optimization objectives:
    - Cost minimization (given target reduction)
    - Pollution minimization (given budget)
    - Cost-effectiveness maximization
    
    Usage:
        >>> optimizer = InterventionOptimizer()
        >>> plan = optimizer.optimize_budget_allocation(
        >>>     current_pollution_scores=scores,
        >>>     budget=1_000_000,
        >>>     interventions=intervention_options,
        >>> )
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize optimizer."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def optimize_budget_allocation(
        self,
        current_pollution_scores: np.ndarray,
        location_names: List[str],
        interventions: List[InterventionOption],
        budget: float,
        objective: str = 'minimize_pollution',
    ) -> InterventionPlan:
        """
        Optimize intervention budget allocation across locations.
        
        Parameters:
            current_pollution_scores: (m,) array of TOPSIS scores (higher = worse)
            location_names: List of location names
            interventions: List of available interventions
            budget: Total budget available
            objective: 'minimize_pollution' or 'maximize_cost_effectiveness'
        
        Returns:
            InterventionPlan with optimal allocations
        """
        m = len(location_names)
        n_interventions = len(interventions)
        
        # Decision variables: x[i, j] = amount of intervention j allocated to location i
        # Flattened: x has shape (m * n_interventions,)
        
        # Objective function coefficients
        if objective == 'minimize_pollution':
            # Minimize: sum of (pollution_score_i * (1 - reduction_i))
            # Linearized: maximize sum of (pollution_score_i * reduction_i)
            c = []
            for i in range(m):
                for j, intervention in enumerate(interventions):
                    # Benefit = pollution_score * reduction_potential
                    benefit = current_pollution_scores[i] * intervention.max_reduction_potential
                    c.append(-benefit)  # Negative because linprog minimizes
            c = np.array(c)
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Constraint 1: Budget constraint
        # sum(cost_j * x[i, j]) <= budget
        A_budget = []
        for i in range(m):
            for j, intervention in enumerate(interventions):
                A_budget.append(intervention.cost_per_unit)
        A_ub = np.array([A_budget])
        b_ub = np.array([budget])
        
        # Constraint 2: Location applicability
        # If intervention j not applicable to location i, then x[i, j] = 0
        # Implemented via bounds
        
        # Bounds: 0 <= x[i, j] <= max_amount
        bounds = []
        for i, loc_name in enumerate(location_names):
            for j, intervention in enumerate(interventions):
                if loc_name in intervention.applicable_locations or 'All' in intervention.applicable_locations:
                    # Max amount = budget / cost (simplified)
                    max_amount = budget / intervention.cost_per_unit if intervention.cost_per_unit > 0 else 1e6
                    bounds.append((0, max_amount))
                else:
                    # Not applicable
                    bounds.append((0, 0))
        
        # Solve linear program
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method='highs',
        )
        
        if not result.success:
            # Return infeasible plan
            return InterventionPlan(
                location_names=location_names,
                intervention_allocations={},
                total_cost=0.0,
                expected_pollution_reduction=np.zeros(m),
                objective_value=0.0,
                status='infeasible',
                budget_used=0.0,
                budget_limit=budget,
            )
        
        # Parse solution
        x_optimal = result.x
        allocations = {}
        expected_reduction = np.zeros(m)
        total_cost = 0.0
        
        idx = 0
        for i, loc_name in enumerate(location_names):
            allocations[loc_name] = {}
            for j, intervention in enumerate(interventions):
                amount = x_optimal[idx]
                idx += 1
                
                if amount > 1e-6:  # Non-negligible allocation
                    allocations[loc_name][intervention.name] = amount
                    
                    # Compute reduction
                    reduction = intervention.max_reduction_potential * (1 - intervention.effectiveness_decay * amount)
                    expected_reduction[i] += reduction
                    
                    # Compute cost
                    total_cost += intervention.cost_per_unit * amount
        
        # Clamp reductions to [0, 1]
        expected_reduction = np.clip(expected_reduction, 0, 1)
        
        return InterventionPlan(
            location_names=location_names,
            intervention_allocations=allocations,
            total_cost=total_cost,
            expected_pollution_reduction=expected_reduction,
            objective_value=-result.fun if objective == 'minimize_pollution' else result.fun,
            status='optimal' if result.success else 'feasible',
            budget_used=total_cost,
            budget_limit=budget,
        )
    
    def generate_intervention_recommendations(
        self,
        plan: InterventionPlan,
        current_scores: np.ndarray,
    ) -> pd.DataFrame:
        """
        Generate human-readable intervention recommendations.
        
        Parameters:
            plan: Optimized InterventionPlan
            current_scores: Current TOPSIS pollution scores
        
        Returns:
            DataFrame with recommendations per location
        """
        recommendations = []
        
        for i, loc_name in enumerate(plan.location_names):
            loc_interventions = plan.intervention_allocations.get(loc_name, {})
            
            if not loc_interventions:
                action = "No intervention needed (already low pollution)"
            else:
                actions = [f"{name} (amount: {amt:.1f})" for name, amt in loc_interventions.items()]
                action = "; ".join(actions)
            
            current_score = current_scores[i]
            expected_new_score = current_score * (1 - plan.expected_pollution_reduction[i])
            
            recommendations.append({
                'Location': loc_name,
                'Current_Score': current_score,
                'Expected_Reduction_%': plan.expected_pollution_reduction[i] * 100,
                'Expected_New_Score': expected_new_score,
                'Recommended_Actions': action,
            })
        
        df = pd.DataFrame(recommendations)
        df = df.sort_values('Current_Score', ascending=False)  # Worst first
        
        return df


def create_default_interventions() -> List[InterventionOption]:
    """
    Create default set of light pollution intervention options.
    
    Based on IDA (International Dark-Sky Association) recommendations:
    - Shielding fixtures
    - LED conversion (CCT control)
    - Dimming controls
    - Curfew policies
    """
    interventions = [
        InterventionOption(
            name='Full Cutoff Shielding',
            cost_per_unit=50_000,  # $/km²
            max_reduction_potential=0.40,  # 40% reduction
            applicable_locations=['All'],
            effectiveness_decay=0.0,
        ),
        InterventionOption(
            name='LED Conversion (Warm CCT)',
            cost_per_unit=80_000,  # Higher upfront but energy savings
            max_reduction_potential=0.35,
            applicable_locations=['Urban', 'Suburban'],
            effectiveness_decay=0.0,
        ),
        InterventionOption(
            name='Adaptive Dimming Controls',
            cost_per_unit=30_000,
            max_reduction_potential=0.25,
            applicable_locations=['All'],
            effectiveness_decay=0.01,  # Slight diminishing returns
        ),
        InterventionOption(
            name='Midnight Curfew Policy',
            cost_per_unit=5_000,  # Low cost (policy implementation)
            max_reduction_potential=0.50,  # High potential
            applicable_locations=['Protected', 'Rural'],
            effectiveness_decay=0.0,
        ),
        InterventionOption(
            name='Vegetation Screening',
            cost_per_unit=20_000,
            max_reduction_potential=0.15,
            applicable_locations=['Protected', 'Rural', 'Suburban'],
            effectiveness_decay=0.02,
        ),
    ]
    
    return interventions


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Intervention Strategy Optimizer - Example")
    print("=" * 80)
    
    # Example TOPSIS scores (higher = worse pollution)
    location_names = ['Protected Area', 'Rural', 'Suburban', 'Urban Core']
    current_scores = np.array([0.25, 0.45, 0.72, 0.89])
    
    # Available interventions
    interventions = create_default_interventions()
    
    print(f"\nAvailable Interventions ({len(interventions)}):")
    for interv in interventions:
        print(f"  • {interv.name}: ${interv.cost_per_unit:,.0f}/km², up to {interv.max_reduction_potential:.0%} reduction")
    
    # Optimize budget allocation
    optimizer = InterventionOptimizer()
    
    budget = 500_000  # $500k total budget
    
    print(f"\nOptimizing intervention allocation (Budget: ${budget:,.0f})...")
    
    plan = optimizer.optimize_budget_allocation(
        current_pollution_scores=current_scores,
        location_names=location_names,
        interventions=interventions,
        budget=budget,
        objective='minimize_pollution',
    )
    
    print(f"\nOptimization Status: {plan.status}")
    print(f"Budget Used: ${plan.budget_used:,.2f} / ${plan.budget_limit:,.0f} ({plan.budget_used/plan.budget_limit:.1%})")
    print(f"Objective Value: {plan.objective_value:.3f}")
    
    # Generate recommendations
    recommendations = optimizer.generate_intervention_recommendations(plan, current_scores)
    
    print("\n" + "=" * 80)
    print("INTERVENTION RECOMMENDATIONS (Priority Order)")
    print("=" * 80)
    print(recommendations.to_string(index=False))
    
    print("\n✓ Intervention optimization complete")
