import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8

df = pd.read_csv('Vol.csv')

voltage_data = df[[f'Bus_{i}' for i in range(1, 34)]].values

voltage_data = voltage_data[:94, :]

hour_1 = np.mean(voltage_data[:2, :], axis=0, keepdims=True)

remaining_data = voltage_data[2:, :]
hours_2_to_24 = np.mean(remaining_data.reshape(23, 4, -1), axis=1)

new_voltage_data = np.vstack((hour_1, hours_2_to_24))

n_steps = new_voltage_data.shape[0]
n_buses = new_voltage_data.shape[1]

time_steps = np.arange(1, n_steps + 1)
bus_ids = np.arange(1, n_buses + 1)
X, Y = np.meshgrid(bus_ids, time_steps)
Z = new_voltage_data

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 0.4))
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Voltage (p.u.)')

ax.set_xlabel('Bus Number')
ax.set_ylabel('Time (Hour)')
ax.set_zlabel('Voltage (p.u.)')
ax.set_title('Bus Voltages over 24 Hours', y=0.85)

ax.set_yticks(np.arange(0, 25, 4))

ax.view_init(elev=20, azim=80)

plt.tight_layout()

plt.savefig('Vol.png', dpi=300, bbox_inches='tight')
plt.show()