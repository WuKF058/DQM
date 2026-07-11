import pandas as pd

dg_df = pd.read_csv('dg.csv')
mg1_df = pd.read_csv('mg1.csv')
n = min(len(mg1_df), len(dg_df))
dg = dg_df.iloc[:n].copy()
mg1 = mg1_df.iloc[:n].copy()

DIESEL_COST_RATE = 0.15
UPSTREAM_PRICE = 0.10
TIME_STEP = 0.25

mg1['diesel_cost'] = mg1['diesel_power'] * DIESEL_COST_RATE * TIME_STEP

def calc_mg_grid_cost(row, dg_row):
    gp = row['grid_power']
    if gp > 0:
        return gp * dg_row['sell_price'] * TIME_STEP
    else:
        return gp * dg_row['buy_price'] * TIME_STEP

mg1['grid_interaction_cost'] = [calc_mg_grid_cost(mg1.iloc[i], dg.iloc[i]) for i in range(n)]
mg1['total_step_cost'] = mg1['diesel_cost'] + mg1['grid_interaction_cost']

dg['revenue_from_mgs'] = dg['grid_power'].clip(lower=0) * dg['sell_price'] * TIME_STEP
dg['cost_to_mgs'] = (-dg['grid_power']).clip(lower=0) * dg['buy_price'] * TIME_STEP
dg['sell_to_upstream'] = (-dg['grid_power']).clip(lower=0)
dg['cost_upstream'] = (dg['upstream_power'] * TIME_STEP * UPSTREAM_PRICE) - (dg['sell_to_upstream'] * TIME_STEP * UPSTREAM_PRICE)
dg['total_step_profit'] = dg['revenue_from_mgs'] - dg['cost_to_mgs'] - dg['cost_upstream']

print(f"MG Total Cost: {mg1['total_step_cost'].sum():.4f}")
print(f"DG Cost: {dg['total_step_profit'].sum():.4f}")