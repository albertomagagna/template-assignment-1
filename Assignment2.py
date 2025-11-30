import gurobipy as gp
from gurobipy import GRB


class Expando(object):
    '''
        A small class which can have attributes set
    '''
    pass


class InputData:  # data.MARKETS, data.generator_supply_steps, data.demand_steps, data.SUPPLY_STEPS, data.DEMAND_STEPS
    
    def __init__(
        self, 
        MARKETS: list,
        GENERATORS: dict[int, dict],
        DEMANDS: dict[int, dict],
    ):
        # List of scenarios
        self.MARKETS = MARKETS
        # Generator supply steps: {1: {'quantity': 100, 'cost': 10}}
        self.generator_supply_steps = GENERATORS
        # Demand steps: {1: {'quantity': 20, 'price': 60}, 2: {'quantity': 90, 'price': 40}}
        self.demand_steps = DEMANDS 
        
        # Keys for quick access
        self.SUPPLY_STEPS = list(self.generator_supply_steps.keys())
        self.DEMAND_STEPS = list(self.demand_steps.keys())



class DAMarketClearing():
    # Adjusted class name for clarity, but maintaining structure
    
    def __init__(self, input_data: InputData):
        self.data = input_data 
        # perfect_information is not relevant for a deterministic market clearing per scenario
        self.variables = Expando()
        self.constraints = Expando() 
        self.results = Expando() 
        self._build_models() 

    def _build_models(self):
        # Build one combined model that contains variables for all scenarios.
        # This allows adding constraints that span scenarios (e.g. total demand across scenarios).
        self.scenario_models = {}
        self.all_results = {}
        # Initialize storage for per-scenario variables and constraints
        self.variables.supply_cleared = {}
        self.variables.demand_cleared = {}
        self.variables.mcp = {}
        self.constraints.supply_cap = {}
        self.constraints.demand_cap = {}
        self.constraints.mcc = {}

        # single combined model
        self.model = gp.Model(name='DA_Market_Clearing_AllScenarios')

        # Accumulate objective terms for all scenarios
        total_objective = 0
        for scenario in self.data.MARKETS:
            self._build_variables(scenario)
            self._build_constraints(scenario)
            total_objective = total_objective + self._build_objective_function(scenario)

        # Global constraint: total cleared demand across all scenarios equals 150
        expr_all = gp.quicksum(
            self.variables.demand_cleared[sc][d]
            for sc in self.data.MARKETS
            for d in list(self.data.demand_steps[sc].keys())
        )
        self.model.addConstr(expr_all == 150, name='Total_Demand_All_Scenarios')

        # set combined objective and store model
        self.model.setObjective(total_objective, GRB.MAXIMIZE)
        self.model.update()
        self.scenario_models['ALL'] = self.model
    
    
    def _build_variables(self, scenario):
        # Production cleared from each supply step for this scenario (market)
        supply_steps = list(self.data.generator_supply_steps[scenario].keys())
        demand_steps = list(self.data.demand_steps[scenario].keys())

        self.variables.supply_cleared[scenario] = {
            s: self.model.addVar(lb=0, ub=GRB.INFINITY, name=f'Supply_Cleared_{s}_{scenario}') for s in supply_steps
        } # 

        # Demand cleared for each demand step for this scenario
        self.variables.demand_cleared[scenario] = {
            d: self.model.addVar(lb=0, ub=GRB.INFINITY, name=f'Demand_Cleared_{d}_{scenario}') for d in demand_steps
        }

        # Market Clearing Price (MCP) - Continuous variable for this scenario
        self.variables.mcp[scenario] = self.model.addVar(lb=0, ub=GRB.INFINITY, name=f'Market_Clearing_Price_{scenario}')

    def _build_constraints(self, scenario):
        # Supply and demand steps for this scenario
        supply_steps = list(self.data.generator_supply_steps[scenario].keys())
        demand_steps = list(self.data.demand_steps[scenario].keys())

        # 1. Supply Capacity Constraints (cannot clear more than offered)
        self.constraints.supply_cap[scenario] = {
            s: self.model.addLConstr(
                self.variables.supply_cleared[scenario][s],
                GRB.LESS_EQUAL,
                self.data.generator_supply_steps[scenario][s]['quantity'],
                name=f'Max_Supply_{s}_{scenario}',
            ) for s in supply_steps
        }

        # 2. Demand Capacity Constraints (cannot clear more than demanded)
        self.constraints.demand_cap[scenario] = {
            d: self.model.addLConstr(
                self.variables.demand_cleared[scenario][d],
                GRB.LESS_EQUAL,
                self.data.demand_steps[scenario][d]['quantity'],
                name=f'Max_Demand_{d}_{scenario}',
            ) for d in demand_steps
        }

        # 3. Market Clearing Condition (MCC): Total Cleared Supply = Total Cleared Demand
        self.constraints.mcc[scenario] = self.model.addLConstr(
            gp.quicksum(self.variables.supply_cleared[scenario][s] for s in supply_steps),
            GRB.EQUAL,
            gp.quicksum(self.variables.demand_cleared[scenario][d] for d in demand_steps),
            name=f'Market_Clearing_Condition_{scenario}',
        )
        # note: total-demand-per-scenario constraint is added separately by
        # `_build_constraint_total_demand` which is called from the model builder.

    def _build_objective_function(self, scenario):
        # Return objective expression for this scenario (so it can be accumulated across scenarios)
        supply_steps = list(self.data.generator_supply_steps[scenario].keys())
        demand_steps = list(self.data.demand_steps[scenario].keys())

        total_demand_value = gp.quicksum(
            self.data.demand_steps[scenario][d]['price'] * self.variables.demand_cleared[scenario][d]
            for d in demand_steps
        )

        total_supply_cost = gp.quicksum(
            self.data.generator_supply_steps[scenario][s]['cost'] * self.variables.supply_cleared[scenario][s]
            for s in supply_steps
        )

        return total_demand_value - total_supply_cost

    def _save_results(self, scenario):
        # Extract variable values for the given scenario and store them
        res = {}
        res['supply_cleared'] = {s: self.variables.supply_cleared[scenario][s].X for s in self.variables.supply_cleared[scenario]}
        res['demand_cleared'] = {d: self.variables.demand_cleared[scenario][d].X for d in self.variables.demand_cleared[scenario]}
        res['mcp'] = self.variables.mcp[scenario].X
        res['objective'] = self.model.ObjVal
        self.all_results[scenario] = res

    def run(self):
        # Optimize the single combined model and save results per scenario
        model = self.scenario_models.get('ALL')
        if model is None:
            raise RuntimeError('No combined model found to run')
        self.model = model
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            for scenario in self.data.MARKETS:
                self._save_results(scenario)
        else:
            raise RuntimeError(f"Optimization of {model.ModelName} was not successful. Status: {self.model.status}")
    
    def plot_results(self, generators=None, demands=None, mcps=None, q_eqs=None, scenarios=None):
        import matplotlib.pyplot as plt
        import numpy as np

        if generators is None:
            generators = {m: list(v.values()) if isinstance(v, dict) else v for m, v in self.data.generator_supply_steps.items()}
        if demands is None:
            demands = {m: list(v.values()) if isinstance(v, dict) else v for m, v in self.data.demand_steps.items()}
        if scenarios is None:
            scenarios = self.data.MARKETS
        if mcps is None:
            mcps = {m: self.all_results.get(m, {}).get('mcp', 0) for m in scenarios}
        if q_eqs is None:
            q_eqs = {m: sum(self.all_results.get(m, {}).get('supply_cleared', {}).values()) for m in scenarios}

        for scenario in scenarios:
            gen = generators[scenario]
            dem = demands[scenario]

            gen_sorted = sorted(gen, key=lambda x: x['cost'])
            gen_quantities = [d['quantity'] for d in gen_sorted]
            gen_costs = [d['cost'] for d in gen_sorted]

            supply_x = np.concatenate(([0], np.cumsum(gen_quantities)))
            supply_y = gen_costs + [gen_costs[-1]]

            dem_sorted = sorted(dem, key=lambda x: x['price'], reverse=True)
            dem_quantities = [d['quantity'] for d in dem_sorted]
            dem_prices = [d['price'] for d in dem_sorted]

            demand_x = np.concatenate(([0], np.cumsum(dem_quantities), np.cumsum(dem_quantities)))
            demand_y = [dem_prices[0]] + dem_prices + [0]

            def supply_price_at(q):
                cum = np.cumsum(gen_quantities)
                idx = np.searchsorted(cum, q, side='right')
                if idx >= len(gen_costs):
                    return gen_costs[-1]
                return gen_costs[idx]

            def demand_price_at(q):
                cum = np.cumsum(dem_quantities)
                idx = np.searchsorted(cum, q, side='right')
                if idx >= len(dem_prices):
                    return dem_prices[-1]
                return dem_prices[idx]

            qpoints = np.unique(np.concatenate(([0], np.cumsum(gen_quantities), np.cumsum(dem_quantities))))
            clearing_q = 0.0
            clearing_p = 0.0
            for q in qpoints:
                sp = supply_price_at(q)
                dp = demand_price_at(q)
                if sp <= dp:
                    clearing_q = q
                    clearing_p = sp

            plt.figure(figsize=(10, 6))
            plt.step(supply_x, supply_y, where='post', label='Supply', color='tab:green', linewidth=2)
            plt.step(demand_x, demand_y, where='pre', label='Demand', color='tab:orange', linewidth=2)
            plt.axvline(x=clearing_q, color='red', linestyle='--', linewidth=1)
            plt.axhline(y=clearing_p, color='red', linestyle='--', linewidth=1)
            plt.plot([clearing_q], [clearing_p], marker='o', color='red')
            plt.legend(loc='best')
            plt.xlabel('Quantity')
            plt.ylabel('Price / Cost')
            if q_eqs.get(scenario, 0) == 0:
                title = f'Supply and Demand Curves for {scenario} (No Market Clearing)'
            else:
                title = f'Supply and Demand Curves for {scenario}\nMarket Clearing Price: ${mcps.get(scenario,0)}$, Quantity: ${q_eqs.get(scenario,0)}$'
            plt.title(title)
            plt.grid(True)
            min_price = min(gen_costs[0], dem_prices[-1]) - 5
            max_price = max(gen_costs[-1], dem_prices[0]) + 5
            plt.ylim(min_price, max_price)
            plt.savefig(f'supply_demand_curve_{scenario}.png')
            plt.close()
    

if __name__ == '__main__':

    MARKETS = ['OIL', 'GAS', 'COAL']

    # Generator Bid: (x: 100, y: 10) -> Supply Step 1: 100 units at Cost 10
    DEMAND_OIL = {1: {'quantity': 100, 'price': 50}}
    DEMAND_COAL = {1: {'quantity': 100, 'price': 100}}
    DEMAND_GAS = {1: {'quantity': 100, 'price': 60}}
    DEMANDS = {
        'OIL': DEMAND_OIL,
        'COAL': DEMAND_COAL,
        'GAS': DEMAND_GAS,
    }

    # Demand Bids: (x: 20, y: 60) and (x: 110, y: 40)
    GENERATOR_OIL = {
        1: {'quantity': 20, 'cost': 60}, 
        2: {'quantity': 90, 'cost': 40},
    }
    GENERATOR_COAL = {
        1: {'quantity': 30, 'cost': 70}, 
        2: {'quantity': 80, 'cost': 50},
    }
    GENERATOR_GAS = {
        1: {'quantity': 25, 'cost': 65}, 
        2: {'quantity': 85, 'cost': 45},
    }
    GENERATORS = {
        'OIL': GENERATOR_OIL,
        'COAL': GENERATOR_COAL,
        'GAS': GENERATOR_GAS,
    }   

    input_data = InputData(
        MARKETS = MARKETS,
        GENERATORS = GENERATORS,
        DEMANDS = DEMANDS,
    )

    model = DAMarketClearing(input_data)
    model.run()
    model.plot_results()
