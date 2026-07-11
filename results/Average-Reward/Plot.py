import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
A = 'FL_PPO'

file_path = f'{A}.xlsx'
df = pd.read_excel(file_path)

episodes = pd.to_numeric(df.iloc[:, 0], errors='coerce').astype(float).values

# labels = ['PPO_VQC_1', 'PPO_VQC_2', 'PPO_VQC_3', 'PPO_DQM_1', 'PPO_DQM_2', 'PPO_DQM_3', 'PPO']
# colors = ['#08306b', '#2879b9', '#73b3d8', '#00441b', '#238b45', '#74c476', '#e76f51']
# markers = ['o', '^', 's', 'D', '*', 'v', 'P']

# labels = ['A2C_VQC', 'A2C', 'A2C_DQM', 'A2C_RBF']
# colors = ['#08306b', '#00441b', '#e76f51', '#f4a261']
# markers = ['o', '^', 's', 'D']

# labels = ['SAC_DQM_1', 'SAC_1', 'SAC_DQM_2', 'SAC_2', 'SAC_VQC']
# colors = ['#00441b', '#08306b', '#238b45', '#2879b9', '#e76f51']
# markers = ['o', '^', 's', 'D', '*']

# labels = ['SAC_DQM', 'SAC']
# colors = ['#00441b', '#08306b']
# markers = ['o', '^']

labels = ['PPO', 'PPO_DQM', 'PPO_VQC']
colors = ['#1e3c72', '#2a9d8f', '#e9c46a']
markers = ['o', '^', 's']

max_ep = int(episodes.max())
target_episodes = np.arange(0, max_ep + 1, 200)
mark_indices = []
for t in target_episodes:
    if t > episodes.max():
        continue
    idx = np.argmin(np.abs(episodes - t))
    mark_indices.append(idx)
mark_indices = sorted(set(mark_indices))

plt.figure(figsize=(12, 5), dpi=300)
window_size = 20

for i in range(len(labels)):
    avg_col_index = i + 1
    raw_col_index = i + 4
    avg_rewards = pd.to_numeric(df.iloc[:, avg_col_index], errors='coerce').astype(float).values
    raw_rewards = pd.to_numeric(df.iloc[:, raw_col_index], errors='coerce').astype(float).values
    rolling_std = pd.Series(raw_rewards).rolling(window=window_size, min_periods=1).std()
    rolling_std = rolling_std.fillna(0).astype(float).values
    color = colors[i]

    plt.plot(episodes, avg_rewards,
             color=color,
             linewidth=2.5,
             alpha=0.9,
             label=labels[i],
             marker=markers[i],
             markersize=8,
             markevery=mark_indices,
             markeredgecolor='none',
             zorder=5)

    plt.fill_between(episodes, avg_rewards - rolling_std, avg_rewards + rolling_std,
                     color=color, alpha=0.1, linewidth=0)

plt.xlabel('Episode')
plt.ylabel('Average Reward')

plt.legend(loc='lower center',
           bbox_to_anchor=(0.5, 1.02),
           framealpha=0.9,
           facecolor='white',
           ncol=5,
           edgecolor='lightgray',
           shadow=True,
           fancybox=True)

plt.grid(False)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
plt.tight_layout()

plt.savefig(f'{A}-Reward.png', dpi=300, bbox_inches='tight')

plt.show()