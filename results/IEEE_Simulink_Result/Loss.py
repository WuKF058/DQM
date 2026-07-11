import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

df_loss = pd.read_csv('Loss.csv')

loss_data = df_loss['Total_Active_Loss_MW'].values

loss_data = loss_data[:94]

hour_1 = np.mean(loss_data[:2])

remaining_data = loss_data[2:]
hours_2_to_24 = np.mean(remaining_data.reshape(23, 4), axis=1)

new_loss_data = np.append(hour_1, hours_2_to_24)

time_steps = np.arange(1, 25)

plt.figure(figsize=(6.5, 2.5), dpi=300)

plt.plot(time_steps, new_loss_data, color='#1e3c72', linewidth=2.5, alpha=0.9)

plt.scatter(time_steps, new_loss_data, marker='o', s=60, color='#1e3c72', alpha=0.85, edgecolors='none', zorder=5)

plt.xlabel('Time (Hour)')
plt.ylabel('Total Active Loss (MW)')

plt.title('Total Active Loss over 24 Hours')

plt.xticks(np.arange(0, 25, 6))
plt.xlim(0.5, 24.5)

plt.grid(False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()

plt.savefig('Loss.png', dpi=300, bbox_inches='tight')

plt.show()