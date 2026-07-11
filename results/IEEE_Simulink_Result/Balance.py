import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_dg = pd.read_csv('dg.csv')
dg_hourly = df_dg[['buy_price', 'sell_price']].groupby(df_dg.index // 4).mean()

all_mg_data = {}
global_p_max = 0
global_p_min = 0

for i in range(1, 5):
    try:
        df = pd.read_csv(f'mg{i}.csv')
    except FileNotFoundError:
        print(f"mg{i}.csv isn't exist! Skipping...")
        continue

    if len(df) == 95:
        pad_row = pd.DataFrame([{col: 0 for col in df.columns}])
        df = pd.concat([pad_row, df], ignore_index=True)
    elif len(df) > 96:
        df = df.iloc[:96]

    col_indices = [4, 5, 6, 7, 8, 13, 14]
    cols_to_use = [df.columns[j] for j in col_indices]
    df_subset = df[cols_to_use].copy()
    df_subset[df.columns[14]] = -df_subset[df.columns[14]]

    df_hourly = df_subset.groupby(df_subset.index // 4).sum()

    all_mg_data[i] = df_hourly

    pos_sum_max = df_hourly[df_hourly > 0].sum(axis=1).max()
    neg_sum_min = df_hourly[df_hourly < 0].sum(axis=1).min()

    if pos_sum_max > global_p_max: global_p_max = pos_sum_max
    if neg_sum_min < global_p_min: global_p_min = neg_sum_min

p_y_max = global_p_max * 1.15
p_y_min = global_p_min * 1.15 if global_p_min < 0 else 0

price_max = dg_hourly.max().max() * 1.1
price_min = dg_hourly.min().min() * 0.9 if dg_hourly.min().min() > 0 else dg_hourly.min().min() * 1.1

npg_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1

for i, df_hourly in all_mg_data.items():
    fig, ax = plt.subplots(figsize=(12, 4))

    df_hourly.plot(kind='bar', stacked=True, ax=ax, width=0.75,
                   color=npg_colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Power (kW)')
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels([f'{h:01d}' for h in range(24)], rotation=0)

    ax.set_ylim(p_y_min, p_y_max)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    ax.axhline(0, color='black', linewidth=1.5)

    ax.spines['top'].set_visible(False)

    ax2 = ax.twinx()

    ax2.plot(np.arange(24), dg_hourly['buy_price'],
             color='#111111', linestyle='-', linewidth=2.5, marker='o', markersize=6, label='Sell Price')
    ax2.plot(np.arange(24), dg_hourly['sell_price'],
             color='#FF8C00', linestyle='--', linewidth=2.5, marker='s', markersize=6, label='Buy Price')

    ax2.set_ylabel('Electricity Price (€/kWh)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_linewidth(1)

    ax2.set_ylim(price_min, price_max)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax.legend(handles1 + handles2, labels1 + labels2,
              loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=5, frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(f'mg{i}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"mg{i}.png is plotted.")