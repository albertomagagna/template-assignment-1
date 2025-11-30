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
        self.variables = Expando()
        self.constraints = Expando() 
        self.results = Expando() 
        self._build_models() 

    def _build_models(self):
        # Build a single combined model so we can add a global constraint across scenarios
        self.all_results = {}
        self.variables.gen_supply = {}
        self.variables.demand = {}

        self.model = gp.Model('AllScenarios')
        total_objective = 0

        # create variables and per-scenario constraints, accumulate objective
        for scenario in self.data.MARKETS:
            self._build_variables(scenario)
            self._build_constraints(scenario)
            total_objective = total_objective + self._build_objective_function(scenario)

        # Global constraint: sum of demand across all scenarios equals 80
        expr_all = gp.quicksum(
            self.variables.demand[sc][d]
            for sc in self.data.MARKETS
            for d in list(self.data.demand_steps[sc].keys())
        )
        self.model.addConstr(expr_all == 100, name='Total_Demand_All_Scenarios')

        # set objective and optimize once
        self.model.setObjective(total_objective, GRB.MAXIMIZE)
        self.model.update()
        self.model.optimize()

        # extract per-scenario results for plotting
        for scenario in self.data.MARKETS:
            if self.model.status == GRB.OPTIMAL:
                res = {}
                res['objective'] = self.model.objVal
                res['gen_supply'] = {s: self.variables.gen_supply[scenario][s].X for s in self.variables.gen_supply[scenario].keys()}
                res['demand'] = {d: self.variables.demand[scenario][d].X for d in self.variables.demand[scenario].keys()}
            else:
                res = {'objective': None, 'gen_supply': {}, 'demand': {}}
            self.all_results[scenario] = res

    def _build_variables(self, scenario):
        # create variables in the combined model and store them per-scenario
        gen_supply_steps = self.data.generator_supply_steps[scenario]
        demand_steps = self.data.demand_steps[scenario]

        self.variables.gen_supply[scenario] = self.model.addVars(gen_supply_steps.keys(), name=f"gen_supply_{scenario}", lb=0, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS)
        self.variables.demand[scenario] = self.model.addVars(demand_steps.keys(), name=f"demand_{scenario}", lb=0, ub=gp.GRB.INFINITY, vtype=GRB.CONTINUOUS)
        self.model.update()
        
    def _build_constraints(self, scenario):
        model = self.model
        gen_supply_steps = self.data.generator_supply_steps[scenario]
        demand_steps = self.data.demand_steps[scenario]

        # Generator Supply Step Constraints (per-scenario)
        self.constraints.gen_supply_limits = model.addConstrs(
            (self.variables.gen_supply[scenario][s] <= gen_supply_steps[s]['quantity'] for s in gen_supply_steps.keys()),
            name=f"gen_supply_limits_{scenario}",
        )

        # Demand Step Constraints (per-scenario)
        self.constraints.demand_limits = model.addConstrs(
            (self.variables.demand[scenario][d] <= demand_steps[d]['quantity'] for d in demand_steps.keys()),
            name=f"demand_limits_{scenario}",
        )

        self.constraints.market_clearing = model.addConstr(
            gp.quicksum(self.variables.gen_supply[scenario][s] for s in gen_supply_steps.keys()) ==
            gp.quicksum(self.variables.demand[scenario][d] for d in demand_steps.keys()),
            name=f"market_clearing_{scenario}",
        )
        model.update()
        
    def _build_objective_function(self, scenario):
        # Return objective expression for the given scenario so the caller can accumulate
        gen_supply_steps = self.data.generator_supply_steps[scenario]
        demand_steps = self.data.demand_steps[scenario]

        total_utility = gp.quicksum(demand_steps[d]['price'] * self.variables.demand[scenario][d] for d in demand_steps.keys())
        total_cost = gp.quicksum(gen_supply_steps[s]['cost'] * self.variables.gen_supply[scenario][s] for s in gen_supply_steps.keys())
        return total_utility - total_cost

    def run(self):
        # Print results collected while building models
        for scenario in self.data.MARKETS:
            res = self.all_results.get(scenario, {})
            print(f"--- Scenario: {scenario} ---")
            print("Objective:", res.get('objective'))
            print("Generator Supply:", res.get('gen_supply'))
            print("Demand:", res.get('demand'))

    def plot_results(self):
        import matplotlib.pyplot as plt

        # Plot supply (offers) as an increasing step function (by ascending cost)
        # and demand as a decreasing step function (by descending price).
        for scenario in self.data.MARKETS:
            supply = self.data.generator_supply_steps[scenario]
            demand = self.data.demand_steps[scenario]

            # Prepare supply: sort by cost ascending (merit order), build cumulative quantities
            supply_items = sorted(supply.items(), key=lambda kv: kv[1]['cost'])
            cum = 0
            supply_cums = []
            supply_costs = []
            for _, v in supply_items:
                cum += v['quantity']
                supply_cums.append(cum)
                supply_costs.append(v['cost'])
            if supply_cums:
                sx = [0] + supply_cums
                sy = supply_costs + [supply_costs[-1]]  # repeat last to match lengths for step('post')
            else:
                sx, sy = [], []

            # Prepare demand: sort by price descending so the step curve is downward
            demand_items = sorted(demand.items(), key=lambda kv: kv[1]['price'], reverse=True)
            cum = 0
            demand_cums = []
            demand_prices = []
            for _, v in demand_items:
                cum += v['quantity']
                demand_cums.append(cum)
                demand_prices.append(v['price'])
            if demand_cums:
                dx = [0] + demand_cums
                dy = demand_prices + [demand_prices[-1]]
            else:
                dx, dy = [], []

            
            # Plot
            plt.figure(figsize=(8, 5))
            if sx:
                plt.step(sx, sy, where='post', label='Generator offers (supply)', color='tab:green')

            # shade cleared supply quantities (if optimization results available)
            cleared_supply = self.all_results.get(scenario, {}).get('gen_supply', {})
            prev = 0
            for (step_key, step) in supply_items:
                q = step['quantity']
                cost = step['cost']
                cleared = cleared_supply.get(step_key, 0)
                # shading range
                x0 = prev
                x1 = prev + min(cleared, q)
                if x1 > x0:
                    plt.fill_between([x0, x1], [0, 0], [cost, cost], color='tab:green', alpha=0.35)
                prev += q

            if dx:
                plt.step(dx, dy, where='post', label='Demand', color='tab:red')

            # shade satisfied demand quantities (if optimization results available)
            cleared_demand = self.all_results.get(scenario, {}).get('demand', {})
            prev = 0
            for (dkey, dstep) in demand_items:
                q = dstep['quantity']
                price = dstep['price']
                cleared = cleared_demand.get(dkey, 0)
                x0 = prev
                x1 = prev + min(cleared, q)
                if x1 > x0:
                    plt.fill_between([x0, x1], [0, 0], [price, price], color='tab:red', alpha=0.25)
                prev += q

            # plot formatting
            plt.ylim(0, max(supply_costs + demand_prices) * 1.1)
            plt.xlabel('Cumulative Quantity')
            plt.ylabel('Price / Cost')
            plt.title(f'Market: {scenario}  (shaded areas = cleared quantities)')
            plt.legend()
            plt.grid(True)
            plt.show()
    
if __name__ == '__main__':

    MARKETS = ['OIL', 'COAL', 'GAS']

    # Generator Bid: (x: 100, y: 10) -> Supply Step 1: 100 units at Cost 10
    DEMAND_OIL = {1: {'quantity': 100, 'price': 50}}
    DEMAND_COAL = {1: {'quantity': 100, 'price': 50}}
    DEMAND_GAS = {1: {'quantity': 100, 'price': 50}}

    DEMANDS = {
        'OIL': DEMAND_OIL,
        'COAL': DEMAND_COAL,
        'GAS': DEMAND_GAS
    }

    # Demand Bids: (x: 20, y: 60) and (x: 110, y: 40)
    GENERATOR_OIL = {
        1: {'quantity': 20, 'cost': 45}, 
        2: {'quantity': 40, 'cost': 40},
    }
    
    GENERATOR_COAL = {
        1: {'quantity': 10, 'cost': 35},
        2: {'quantity': 50, 'cost': 80},
    }

    GENERATOR_GAS = {
        1: {'quantity': 40, 'cost': 25},
        2: {'quantity': 50, 'cost': 100},
    }
   
    GENERATORS = {
        'OIL': GENERATOR_OIL,
        'COAL': GENERATOR_COAL,
        'GAS': GENERATOR_GAS
    }   

    input_data = InputData(
        MARKETS = MARKETS,
        GENERATORS = GENERATORS,
        DEMANDS = DEMANDS,
    )

    model = DAMarketClearing(input_data)
    model.run()
    model.plot_results()

