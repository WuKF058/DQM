import random

import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Dict, MultiDiscrete, Discrete
from typing import Dict as TypingDict, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from gym import Env
from gym.spaces import MultiDiscrete, Box


class DistributionGridEnv(Env):
    """
      - grid_power gp > 0 : DG sell the power to MG
      - grid_power gp < 0 : DG buy the power to MG
    So：
      buy_power  = max(gp, 0)   (MG buy, DG sell)
      sell_power = max(-gp, 0)  (MG sell, DG buy)
    """

    def __init__(
            self,
            init_buy_price: float,
            init_sell_price: float,
            price_step: float = 0.005,
            price_min: float = 0.07,
            price_max: float = 0.13,
            min_spread: float = 0.03,
            max_capacity: float = 15.0,
            capacity_noise: float = 0.0,
            upstream_price: float = 0.10,
            time_step: float = 0.25,
            simulation_steps: int = 100,
            seed: int | None = None,
    ):
        super().__init__()
        self.action_space = MultiDiscrete([3, 3])
        self.observation_space = Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        self.init_buy_price = float(init_buy_price)
        self.init_sell_price = float(init_sell_price)
        self.buy_price = float(init_buy_price)
        self.sell_price = float(init_sell_price)

        self.price_step = float(price_step)
        self.price_min = float(price_min)
        self.price_max = float(price_max)
        self.min_spread = float(min_spread)

        self.max_capacity = float(max_capacity)
        self.capacity_noise = float(capacity_noise)
        self.upstream_price = float(upstream_price)
        self.time_step = float(time_step)

        self.simulation_steps = int(simulation_steps)
        self.current_step = 0
        self.total_profit = 0.0

        self.rng = np.random.default_rng(seed)
        self.available_capacity = self._sample_capacity()

        self._grid_power = 0.0

        self._enforce_spread()
        if self.upstream_price >= self.price_max:
            raise ValueError("upstream_price Shouldn't >= price_max!")

    def set_grid_power(self, grid_power: float):
        self._grid_power = float(grid_power)

    def _sample_capacity(self) -> float:
        cap = self.max_capacity
        if self.capacity_noise > 0:
            cap += self.rng.normal(0.0, self.capacity_noise)
        return float(np.clip(cap, 0.0, self.max_capacity))

    def _enforce_spread(self):
        self.buy_price = float(np.clip(self.buy_price, self.price_min, self.price_max))
        self.sell_price = float(np.clip(self.sell_price, self.price_min, self.price_max))
        if self.sell_price < self.buy_price + self.min_spread:
            self.sell_price = min(self.buy_price + self.min_spread, self.price_max)
            if self.sell_price < self.buy_price + self.min_spread:
                self.buy_price = max(self.sell_price - self.min_spread, self.price_min)

    def _get_obs(self) -> np.ndarray:
        buy_norm = (self.buy_price - self.price_min) / (self.price_max - self.price_min)
        sell_norm = (self.sell_price - self.price_min) / (self.price_max - self.price_min)
        cap_norm = 0.0 if self.max_capacity <= 0 else (self.available_capacity / self.max_capacity)
        return np.array([buy_norm, sell_norm, cap_norm], dtype=np.float32)

    def apply_price_action(self, action):
        """only update the price, do not update reward and time step。"""
        a = np.asarray(action, dtype=np.int64).reshape(-1)
        if a.shape[0] != 2:
            raise ValueError(f"DG action must be 2-dim, got {a.shape}")
        a = np.clip(a, 0, np.array([2, 2], dtype=np.int64))
        buy_dir = int(a[0]) - 1
        sell_dir = int(a[1]) - 1

        if buy_dir != 0:
            self.buy_price = float(np.clip(self.buy_price + buy_dir * self.price_step, self.price_min, self.price_max))
        if sell_dir != 0:
            self.sell_price = float(
                np.clip(self.sell_price + sell_dir * self.price_step, self.price_min, self.price_max))

        self._enforce_spread()
        return self._get_obs()

    def settle(self, grid_power: float):
        """
        use now grid_power calculate the profit and update the time step。
        Return：obs_next, reward, done, info
        """
        gp = float(grid_power)

        buy_power = max(gp, 0.0)  # kW
        sell_power = max(-gp, 0.0)  # kW

        # DG has own available_capacity (used to provide power to MG), when the power is not enough, DG will buy some from the upper grid
        upstream_power = max(buy_power - self.available_capacity, 0.0)

        buy_energy = buy_power * self.time_step  # kWh
        sell_energy = sell_power * self.time_step  # kWh
        upstream_energy = upstream_power * self.time_step

        # profit
        revenue_from_mg = buy_energy * self.sell_price
        cost_to_mg = sell_energy * self.buy_price
        cost_upstream = upstream_energy * self.upstream_price

        reward = revenue_from_mg - cost_to_mg - cost_upstream
        self.total_profit += reward

        self.current_step += 1
        done = self.current_step >= self.simulation_steps - 1

        self.available_capacity = self._sample_capacity()

        obs_next = self._get_obs()
        info = {
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "grid_power_used": gp,
            "upstream_power": upstream_power,
            "buy_power": buy_power,
            "sell_power": sell_power,
            "profit": reward,
            "total_profit": self.total_profit,
        }
        return obs_next, float(reward), bool(done), info

    def step(self, action):
        self.apply_price_action(action)
        return self.settle(getattr(self, "_grid_power", 0.0))

    def reset(self):
        self.buy_price = float(self.init_buy_price)
        self.sell_price = float(self.init_sell_price)
        self.current_step = 0
        self.total_profit = 0.0
        self.available_capacity = self._sample_capacity()
        self._grid_power = 0.0
        self._enforce_spread()
        return self._get_obs()


class MicrogridEnv(Env):

    def __init__(self, data_path: str, config: TypingDict[str, Any], Start: int):
        super().__init__()
        start_row = 1 + Start * 192
        skip = range(1, start_row)

        self.data = pd.read_csv(data_path, header=0, skiprows=skip, nrows=100)
        self.data['time'] = pd.to_datetime(self.data['time'])
        self.data = self.data.set_index('time')

        self.config = config
        self.simulation_steps = len(self.data)
        self.current_step = 0

        self.time_step = config.get("time_step", 0.25)  # 15 minutes = 0.25 hours

        self.pv_capacity = config.get("pv_capacity", 100)
        self.wind_capacity = config.get("wind_capacity", 50)
        self.diesel_capacity = config.get("diesel_capacity", 200)
        self.battery_capacity = config.get("battery_capacity", 500)
        self.battery_soc_min = config.get("battery_soc_min", 0.2)
        self.battery_soc_max = config.get("battery_soc_max", 0.95)
        self.battery_efficiency = config.get("battery_efficiency", 0.95)
        self.load_demand_scale = config.get("load_demand_scale", 3.5)
        self.battery_max_power_ratio = config.get("battery_max_power_ratio", 0.4)

        self.diesel_cost = config.get("diesel_cost", 0.15)
        self.grid_buy_price = config.get("grid_buy_price", 0.12)
        self.grid_sell_price = config.get("grid_sell_price", 0.08)
        self.curtailment_penalty = config.get("curtailment_penalty", 0.5)
        self.load_shedding_penalty = config.get("load_shedding_penalty", 2.0)

        self.line_capacity_pv = config.get("line_capacity_pv", 120)
        self.line_capacity_wind = config.get("line_capacity_wind", 60)
        self.line_capacity_diesel = config.get("line_capacity_diesel", 250)
        self.line_capacity_battery = config.get("line_capacity_battery", 300)
        self.line_capacity_grid = config.get("line_capacity_grid", 400)
        self.line_capacity_load = config.get("line_capacity_load", 500)
        self.line_safety_margin = config.get("line_safety_margin", 0.9)
        self.line_violation_penalty_coef = config.get("line_violation_penalty_coef", 15.0)

        self.dynamic_curtailment_penalty = config.get("dynamic_curtailment_penalty", True)

        self.battery_soc_penalty_high = config.get("battery_soc_penalty_high", 0.85)
        self.battery_soc_penalty_low = config.get("battery_soc_penalty_low", 0.25)

        self.diesel_fuel_capacity = config.get("diesel_fuel_capacity", 1000)
        self.fuel_consumption_rate = config.get("fuel_consumption_rate", 0.1)

        self.battery_soc = 0.5
        self.diesel_fuel_level = 100.0
        self.cumulative_cost = 0.0
        self.renewable_utilization = 0.0

        self.line_violations = []
        self.total_line_violations = 0
        self.max_line_violation = 0.0

        # 0: the state of SOC {-1:discharge, 0:free, 1:charge}
        # 1: the state of fuel {-1:on, 0:free, 1:off}

        self.action_space = MultiDiscrete([3, 3])


        self.observation_space = Dict({
            'renewable_generation': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'load_demand': Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'battery_status': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'grid_price': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'time_info': Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'fuel_level': Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'power_balance_error': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })

        self.operation_history = []

        self.balance_errors = []
        self.max_balance_error = 0.0
        self.average_balance_error = 0.0

        self.episode_rewards = []
        self.episode_costs = []

        self._validate_data()
        self.reset()

        print(f"  - PV={self.pv_capacity}kW, Wind={self.wind_capacity}kW, Fuel={self.diesel_capacity}kW")
        print(f"  - {self.battery_capacity}kWh, SOC: {self.battery_soc_min}-{self.battery_soc_max}")
        print(f"  - Fuel Capacity: {self.diesel_fuel_capacity}kWh")
        print(f"  - Timestep: {self.time_step} hours ({self.time_step * 60} minutes)")
        print(
            f"  - Line Capacity: PV={self.line_capacity_pv}kW, Wind={self.line_capacity_wind}kW, Fuel={self.line_capacity_diesel}kW")
        print(f"  - Action Space: 2D-Discrete-Control (MultiDiscrete([3,3]), {-1, 0, 1})")
        print(f"  - State Space: {self._get_state_dimension()}D feature")

    def set_grid_price(self, buy_price: float, sell_price: float):
        self.grid_buy_price = float(buy_price)
        self.grid_sell_price = float(sell_price)

    def _validate_data(self):
        required_columns = ['solar_power', 'wind_power', 'household_power', 'EUR/kWh']
        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Data doesn't have: {missing_columns}")

    def _get_state_dimension(self):
        total_dim = 0
        for space in self.observation_space.spaces.values():
            if hasattr(space, 'shape'):
                total_dim += np.prod(space.shape)
        return total_dim

    def _check_action_bounds(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)

        if action.shape[0] != 2:
            raise ValueError(f"Error of action shape: Expected Shape is 2D(soc_state, diesel_state), Real Shape={action.shape}")

        # MultiDiscrete: Every Shape get {0,1,2}, then project to {-1,0,1}
        if not self.action_space.contains(action):
            action = np.clip(action, 0, self.action_space.nvec - 1)

        # 0->-1, 1->0, 2->1
        mapped = action - 1
        return mapped.astype(np.int64)

    def _get_time_features(self, timestamp) -> np.ndarray:
        hour = timestamp.hour / 24.0
        month = timestamp.month / 12.0
        return np.array([hour, month], dtype=np.float32)

    def _normalize_state(self, raw_state: dict) -> dict:
        normalized = {}

        pv_norm = raw_state['pv_generation'] / self.pv_capacity
        wind_norm = raw_state['wind_generation'] / self.wind_capacity
        normalized['renewable_generation'] = np.array([pv_norm, wind_norm], dtype=np.float32)

        max_load = self.pv_capacity + self.wind_capacity + self.diesel_capacity
        normalized['load_demand'] = np.array([raw_state['load_demand'] / max_load], dtype=np.float32)

        normalized['battery_status'] = np.array([
            self.battery_soc,
            raw_state['battery_power'] / (self.battery_capacity * self.battery_max_power_ratio)
        ], dtype=np.float32)

        normalized['grid_price'] = np.array(
            [self.grid_buy_price / 0.2, self.grid_sell_price / 0.2],
            dtype=np.float32
        )

        normalized['time_info'] = raw_state['time_info']

        normalized['fuel_level'] = np.array([self.diesel_fuel_level / 100.0], dtype=np.float32)

        normalized['power_balance_error'] = np.array([raw_state['power_balance_error'] / 50.0], dtype=np.float32)

        return normalized

    def _get_observation(self) -> dict:
        timestamp = self.data.index[self.current_step]
        raw_state = {
            'pv_generation': self.data.iloc[self.current_step]['solar_power'],
            'wind_generation': self.data.iloc[self.current_step]['wind_power'],
            'load_demand': self.data.iloc[self.current_step]['household_power'] * self.load_demand_scale,

            'grid_price': 0.0,

            'battery_power': 0.0,
            'time_info': self._get_time_features(timestamp),
            'power_balance_error': 0.0,
        }
        return self._normalize_state(raw_state)

    def reset(self) -> dict:
        self.current_step = 0
        self.battery_soc = 0.5
        self.diesel_fuel_level = 100.0
        self.cumulative_cost = 0.0
        self.renewable_utilization = 0.0
        self.operation_history = []

        self.balance_errors = []
        self.max_balance_error = 0.0
        self.average_balance_error = 0.0
        self.episode_rewards = []
        self.episode_costs = []

        self.line_violations = []
        self.total_line_violations = 0
        self.max_line_violation = 0.0

        return self._get_observation()

    def _check_line_constraints(self, power_flows: TypingDict[str, float]) -> Tuple[bool, TypingDict[str, TypingDict]]:
        """
        P² + Q² ≤ S_max²
        |P| ≤ S_max

        Args:
            power_flows:

        Returns:
            Is Out of Limit?
        """
        violations = {}
        has_violation = False

        lines = {
            'pv_line': power_flows.get('pv_power', 0),
            'wind_line': power_flows.get('wind_power', 0),
            'diesel_line': power_flows.get('diesel_power', 0),
            'battery_line': power_flows.get('battery_power', 0),
            'grid_line': power_flows.get('grid_power', 0),
            'load_line': power_flows.get('load_demand', 0)
        }

        line_capacities = {
            'pv_line': self.line_capacity_pv * self.line_safety_margin,
            'wind_line': self.line_capacity_wind * self.line_safety_margin,
            'diesel_line': self.line_capacity_diesel * self.line_safety_margin,
            'battery_line': self.line_capacity_battery * self.line_safety_margin,
            'grid_line': self.line_capacity_grid * self.line_safety_margin,
            'load_line': self.line_capacity_load * self.line_safety_margin
        }

        for line_name, power in lines.items():
            apparent_power = abs(power)
            max_capacity = line_capacities[line_name]

            if apparent_power > max_capacity:
                has_violation = True
                violation_amount = apparent_power - max_capacity
                violation_percentage = (violation_amount / max_capacity) * 100

                violations[line_name] = {
                    'actual_power': power,
                    'apparent_power': apparent_power,
                    'capacity': max_capacity,
                    'violation_amount': violation_amount,
                    'violation_percentage': violation_percentage,
                    'safety_margin_violated': apparent_power > line_capacities[line_name]
                }

                self.max_line_violation = max(self.max_line_violation, violation_amount)
                self.total_line_violations += 1

        if violations:
            self.line_violations.append({
                'timestamp': self.data.index[self.current_step],
                'violations': violations
            })

        return has_violation, violations

    def _calculate_power_balance(self, action: np.ndarray) -> dict:
        """
        action: np.ndarray, shape=(2,)
            action[0] = soc_state   ∈ {-1,0,1}  (-1 discharge, 0 free, 1 charge)
            action[1] = diesel_state∈ {-1,0,1}  (-1 on , 0 free, 1 off)

        """
        pv_gen = float(self.data.iloc[self.current_step]['solar_power'])
        wind_gen = float(self.data.iloc[self.current_step]['wind_power'])
        load_demand = float(self.data.iloc[self.current_step]['household_power'] * self.load_demand_scale)
        total_renewable = pv_gen + wind_gen

        Balance = total_renewable - load_demand
        soc_state = int(action[0])
        diesel_state = int(action[1])

        max_battery_power_raw = self.battery_capacity * self.battery_max_power_ratio
        max_battery_power = min(max_battery_power_raw, self.line_capacity_battery * self.line_safety_margin)

        battery_power = 0.0
        if soc_state == 1:
            available_charge_capacity = max(0.0, (self.battery_soc_max - self.battery_soc) * self.battery_capacity)
            soc_limited_charge_power = available_charge_capacity / self.time_step  # kW

            if Balance > 0:
                feasible_charge_power = min(Balance, max_battery_power, soc_limited_charge_power)
                battery_power = -max(0.0, feasible_charge_power)
            else:
                battery_power = 0.0

        elif soc_state == -1:
            available_discharge_capacity = max(0.0,
                                               (self.battery_soc - self.battery_soc_min) * self.battery_capacity)  # kWh
            soc_limited_discharge_power = available_discharge_capacity / self.time_step  # kW

            if Balance < 0:
                feasible_discharge_power = min(-Balance, max_battery_power, soc_limited_discharge_power)
                battery_power = max(0.0, feasible_discharge_power)
            else:
                battery_power = 0.0

        else:
            battery_power = 0.0

        max_buy_grid_power = self.line_capacity_grid * self.line_safety_margin
        max_sell_grid_power = self.line_capacity_grid * self.line_safety_margin
        grid_power = 0.0

        net = total_renewable + battery_power - load_demand

        if net < 0:
            grid_power = min(-net, max_buy_grid_power)
            net += grid_power
        elif net > 0:
            grid_power = -min(net, max_sell_grid_power)
            net += grid_power

        diesel_power = 0.0

        if diesel_state == -1 and net < 0:
            deficit = -net

            diesel_cap = min(self.diesel_capacity, self.line_capacity_diesel * self.line_safety_margin)

            available_fuel = (self.diesel_fuel_level / 100.0) * self.diesel_fuel_capacity
            max_by_fuel = (available_fuel / (self.time_step * self.fuel_consumption_rate)
                           if self.fuel_consumption_rate > 0 else diesel_cap)
            diesel_cap = max(0.0, min(diesel_cap, max_by_fuel))
            diesel_power = min(deficit, diesel_cap)
            net += diesel_power

        load_shedding = 0.0
        renewable_curtailment = 0.0

        if net < 0:
            load_shedding = min(load_demand, -net)
            net += load_shedding
        elif net > 0:
            renewable_curtailment = min(total_renewable, net)
            net -= renewable_curtailment

        load_shedding = float(np.clip(load_shedding, 0.0, load_demand))
        renewable_curtailment = float(np.clip(renewable_curtailment, 0.0, total_renewable))

        def allocate_curtailment_by_line(pv_gen_, wind_gen_, total_curtailment_):
            if total_curtailment_ <= 0:
                return 0.0, 0.0

            pv_line_capacity = self.line_capacity_pv * self.line_safety_margin
            wind_line_capacity = self.line_capacity_wind * self.line_safety_margin

            pv_c = 0.0
            wind_c = 0.0

            if pv_gen_ > pv_line_capacity:
                pv_c = min(total_curtailment_, pv_gen_ - pv_line_capacity)

            remaining = total_curtailment_ - pv_c
            if remaining > 0 and wind_gen_ > wind_line_capacity:
                wind_c = min(remaining, wind_gen_ - wind_line_capacity)

            remaining = total_curtailment_ - (pv_c + wind_c)
            if remaining > 0:
                total_r = pv_gen_ + wind_gen_
                if total_r > 0:
                    pv_c += remaining * (pv_gen_ / total_r)
                    wind_c += remaining * (wind_gen_ / total_r)
            return float(pv_c), float(wind_c)

        pv_curtailment, wind_curtailment = allocate_curtailment_by_line(pv_gen, wind_gen, renewable_curtailment)

        power_flows = {
            'pv_power': pv_gen - pv_curtailment,
            'wind_power': wind_gen - wind_curtailment,
            'diesel_power': diesel_power,
            'battery_power': battery_power,
            'grid_power': grid_power,
            'load_demand': load_demand - load_shedding
        }

        has_line_violation, line_violations = self._check_line_constraints(power_flows)

        final_generation = (pv_gen - pv_curtailment) + (
                wind_gen - wind_curtailment) + diesel_power + battery_power + grid_power
        final_consumption = load_demand - load_shedding
        verified_balance = final_generation - final_consumption

        return {
            'diesel_power': float(diesel_power),
            'battery_power': float(battery_power),
            'grid_power': float(grid_power),
            'load_shedding': float(load_shedding),
            'renewable_curtailment': float(renewable_curtailment),
            'pv_curtailment': float(pv_curtailment),
            'wind_curtailment': float(wind_curtailment),
            'power_balance': float(verified_balance),
            'total_generation': float(final_generation),
            'total_consumption': float(final_consumption),
            'renewable_generation': float(total_renewable),
            'load_demand': float(load_demand),
            'soc_state': soc_state,
            'diesel_state': diesel_state,
            'has_line_violation': has_line_violation,
            'line_violations': line_violations,
            'power_flows': power_flows,
            'Balance': Balance
        }

    def _calculate_reward(self, power_balance: dict) -> float:
        """简化奖励函数"""
        diesel_cost = power_balance['diesel_power'] * self.diesel_cost * self.time_step

        grid_cost = 0
        grid_power = power_balance['grid_power']
        if grid_power > 0:
            grid_cost = grid_power * self.grid_buy_price * self.time_step
        elif grid_power < 0:
            grid_cost = grid_power * self.grid_sell_price * self.time_step

        total_penalty = 0
        total_penalty += power_balance['load_shedding'] * self.load_shedding_penalty * 10.0
        if power_balance['has_line_violation']:
            for violation_info in power_balance['line_violations'].values():
                total_penalty += violation_info['violation_amount'] * self.line_violation_penalty_coef

        soc_reward = 0
        soc = self.battery_soc
        if 0.3 <= soc <= 0.8:
            soc_reward = 20.0
        # if power_balance['Balance'] < 0 and power_balance['soc_state'] != -1:
        #     total_penalty += 10.0
        # elif power_balance['Balance'] > 0 and power_balance['soc_state'] != 1:
        #     total_penalty += 10.0
        reward = -(diesel_cost + grid_cost) + soc_reward - total_penalty
        # if self.current_step >= self.simulation_steps * 0.8:
        #     soc_err = self.battery_soc - 0.5
        #     progress = self.current_step / max(1, (self.simulation_steps - 1))  # 0->1
        #     lambda_schedule = self.config.get("soc_schedule_penalty", 1.0)
        #     reward -= lambda_schedule * (progress ** 4) * (soc_err ** 2)
        # if self.current_step == self.simulation_steps - 1:
        #     soc_err = self.battery_soc - 0.5
        #     lambda_schedule = self.config.get("soc_schedule_penalty", 10.0)
        #     reward -= lambda_schedule * (soc_err ** 2)
        return np.clip(reward, -100, 100)

    def _update_battery_soc(self, battery_power: float):
        if battery_power < 0:
            energy_change = -battery_power * self.time_step * self.battery_efficiency
            soc_change = energy_change / self.battery_capacity
            new_soc = self.battery_soc + soc_change
            self.battery_soc = max(self.battery_soc_min, min(self.battery_soc_max, new_soc))
        else:
            energy_change = -battery_power * self.time_step / self.battery_efficiency
            soc_change = energy_change / self.battery_capacity
            new_soc = self.battery_soc + soc_change
            self.battery_soc = min(max(self.battery_soc_min, new_soc), self.battery_soc_max)

    def _update_fuel_level(self, diesel_power: float):
        if diesel_power > 0:
            fuel_consumed = diesel_power * self.time_step * self.fuel_consumption_rate
            fuel_percentage_consumed = (fuel_consumed / self.diesel_fuel_capacity) * 100
            self.diesel_fuel_level = max(0, self.diesel_fuel_level - fuel_percentage_consumed)

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        action = self._check_action_bounds(action)

        power_balance = self._calculate_power_balance(action)

        reward = self._calculate_reward(power_balance)

        self._update_battery_soc(power_balance['battery_power'])
        self._update_fuel_level(power_balance['diesel_power'])

        self.cumulative_cost += -reward

        balance_error = power_balance['power_balance']
        self.balance_errors.append(abs(balance_error))
        self.max_balance_error = max(self.max_balance_error, abs(balance_error))
        self.average_balance_error = np.mean(self.balance_errors)

        self.episode_rewards.append(reward)
        self.episode_costs.append(-reward)

        operation_record = {
            'timestamp': self.data.index[self.current_step],
            'action': action.tolist(),
            'reward': reward,
            'battery_soc': self.battery_soc,
            'diesel_power': power_balance['diesel_power'],
            'battery_power': power_balance['battery_power'],
            'grid_power': power_balance['grid_power'],
            'load_shedding': power_balance['load_shedding'],
            'renewable_curtailment': power_balance['renewable_curtailment'],
            'fuel_level': self.diesel_fuel_level,
            'power_balance_error': balance_error,
            'total_generation': power_balance['total_generation'],
            'total_consumption': power_balance['total_consumption'],
            'renewable_generation': power_balance['renewable_generation'],
            'load_demand': power_balance['load_demand'],
            'has_line_violation': power_balance['has_line_violation'],
            'line_violations': power_balance.get('line_violations', {})
        }
        self.operation_history.append(operation_record)

        self.current_step += 1
        done = self.current_step >= self.simulation_steps - 1

        observation = self._get_observation()

        observation['battery_status'][1] = power_balance['battery_power'] / (
                self.battery_capacity * self.battery_max_power_ratio)

        observation['power_balance_error'][0] = np.clip(balance_error / 50.0, -1, 1)

        info = {
            'power_balance': power_balance,
            'cumulative_cost': self.cumulative_cost,
            'battery_soc': self.battery_soc,
            'fuel_level': self.diesel_fuel_level,
            'balance_error': balance_error,
            'max_balance_error': self.max_balance_error,
            'average_balance_error': self.average_balance_error,
            'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'episode_cost_mean': np.mean(self.episode_costs) if self.episode_costs else 0,
            'renewable_utilization': (power_balance['renewable_generation'] -
                                      power_balance['renewable_curtailment']) /
                                     power_balance['renewable_generation'] if power_balance[
                                                                                  'renewable_generation'] > 0 else 0,
            'load_supply_reliability': 1 - (power_balance['load_shedding'] / power_balance['load_demand'])
            if power_balance['load_demand'] > 0 else 1,
            'has_line_violation': power_balance['has_line_violation'],
            'line_violations_count': len(power_balance.get('line_violations', {})),
            'total_line_violations': self.total_line_violations,
            'max_line_violation': self.max_line_violation
        }

        if self.current_step % 100 == 0:
            print(f"\nStep {self.current_step} 摘要:")
            print(f"  SOC: {self.battery_soc:.4f}")
            print(f"  Balance Error: {balance_error:.3f} kW")
            print(f"  Average Error: {self.average_balance_error:.3f} kW")
            print(f"  The time of line_violations: {self.total_line_violations}")

        if done:
            print(f"\n{'=' * 50}")
            print(f"Episode 完成:")
            print(f"  Total Step: {self.current_step}")
            print(f"  Final SOC: {self.battery_soc:.4f}")
            print(f"  Total Cost: €{self.cumulative_cost:.2f}")
            print(f"  Averate Reward: {np.mean(self.episode_rewards):.3f}")
            print(f"  Max Balance Error: {self.max_balance_error:.3f} kW")
            print(f"  Average Balance Error: {self.average_balance_error:.3f} kW")
            print(f"  The time of line_violations: {self.total_line_violations}")
            print(f"  The maximum line_violation: {self.max_line_violation:.3f} kW")
            print(f"{'=' * 50}\n")

            self.balance_errors = []
            self.episode_rewards = []
            self.episode_costs = []
            self.max_balance_error = 0.0
            self.average_balance_error = 0.0
            self.line_violations = []
            self.total_line_violations = 0
            self.max_line_violation = 0.0

        return observation, reward, done, info

    def render(self, mode: str = 'human'):
        if len(self.operation_history) == 0:
            print("No operation history to render.")
            return

        print(f"Step {self.current_step}:")
        print(f"  Battery SOC: {self.battery_soc:.2%}")
        print(f"  Diesel Fuel: {self.diesel_fuel_level:.1f}%")
        print(f"  Cumulative Cost: €{self.cumulative_cost:.2f}")

        if self.line_violations:
            print(f"  Line Violations: {self.total_line_violations}")

    def get_performance_metrics(self) -> dict:
        if len(self.operation_history) == 0:
            return {}

        history_df = pd.DataFrame(self.operation_history)

        total_load_shedding = history_df['load_shedding'].sum() / 4
        total_curtailment = history_df['renewable_curtailment'].sum() / 4
        total_diesel_energy = history_df['diesel_power'].sum() / 4
        total_renewable_energy = (self.data['solar_power'] + self.data['wind_power']).sum() / 4

        renewable_utilization = (
            (total_renewable_energy - total_curtailment) / total_renewable_energy
            if total_renewable_energy > 0 else 0
        )
        load_supply_reliability = (
            1 - (total_load_shedding / (self.data['household_power'].sum() / 4))
            if self.data['household_power'].sum() > 0 else 1
        )

        return {
            'total_cost': self.cumulative_cost,
            'renewable_utilization': renewable_utilization,
            'load_supply_reliability': load_supply_reliability,
            'total_load_shedding': total_load_shedding,
            'total_renewable_curtailment': total_curtailment,
            'total_diesel_energy': total_diesel_energy,
            'average_fuel_level': history_df['fuel_level'].mean(),
            'total_line_violations': self.total_line_violations,
            'max_line_violation': self.max_line_violation,
            'line_violation_frequency': self.total_line_violations / len(history_df) if len(history_df) > 0 else 0
        }

    def plot_operation(self):
        if len(self.operation_history) < 2:
            print("Not enough data to plot.")
            return

        history_df = pd.DataFrame(self.operation_history)
        history_df.set_index('timestamp', inplace=True)

        has_line_violations = 'has_line_violation' in history_df.columns

        if has_line_violations:
            rows = 4
            fig = make_subplots(
                rows=rows, cols=2,
                subplot_titles=(
                    'Power Balance', 'Battery SOC',
                    'Diesel Power', 'Grid Interaction',
                    'Load Shedding', 'Renewable Curtailment',
                    'Line Violation Status', 'Line Violation Magnitude'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
        else:
            rows = 3
            fig = make_subplots(
                rows=rows, cols=2,
                subplot_titles=(
                    'Power Balance', 'Battery SOC',
                    'Diesel Power', 'Grid Interaction',
                    'Load Shedding', 'Renewable Curtailment'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

        renewable_power = self.data['solar_power'] + self.data['wind_power']
        fig.add_trace(go.Scatter(x=history_df.index, y=renewable_power[:len(history_df)],
                                 name="Renewable", line=dict(color='green')), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=history_df.index, y=self.data['household_power'][:len(history_df)] * self.load_demand_scale,
                       name="Load", line=dict(color='red')), row=1, col=1)

        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['battery_soc'],
                                 name="Battery SOC", line=dict(color='blue')), row=1, col=2)

        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['diesel_power'],
                                 name="Diesel Power", line=dict(color='orange')), row=2, col=1)

        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['grid_power'],
                                 name="Grid Power", line=dict(color='purple')), row=2, col=2)

        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['load_shedding'],
                                 name="Load Shedding", line=dict(color='red')), row=3, col=1)

        fig.add_trace(go.Scatter(x=history_df.index, y=history_df['renewable_curtailment'],
                                 name="Renewable Curtailment", line=dict(color='brown')), row=3, col=2)

        if has_line_violations:
            violation_status = history_df['has_line_violation'].astype(int)
            fig.add_trace(go.Scatter(x=history_df.index, y=violation_status,
                                     name="Line Violation", line=dict(color='red'), mode='markers'),
                          row=4, col=1)

            if 'line_violations' in history_df.columns:
                total_violations = []
                for violations in history_df['line_violations']:
                    if isinstance(violations, dict) and violations:
                        total = sum(v['violation_amount'] for v in violations.values())
                        total_violations.append(total)
                    else:
                        total_violations.append(0)

                fig.add_trace(go.Scatter(x=history_df.index, y=total_violations,
                                         name="Violation Amount", line=dict(color='darkred')),
                              row=4, col=2)

        fig.update_layout(height=1000 if has_line_violations else 800,
                          title_text="Microgrid Operation Analysis (Fixed Version)")
        return fig

    def get_line_statistics(self) -> dict:
        if not self.line_violations:
            return {}

        line_violation_counts = {}
        max_violation_by_line = {}
        total_violation_by_line = {}

        for record in self.line_violations:
            for line_name, violation_info in record['violations'].items():
                line_violation_counts[line_name] = line_violation_counts.get(line_name, 0) + 1

                if line_name not in max_violation_by_line or violation_info['violation_amount'] > max_violation_by_line[
                    line_name]:
                    max_violation_by_line[line_name] = violation_info['violation_amount']

                total_violation_by_line[line_name] = total_violation_by_line.get(line_name, 0) + violation_info[
                    'violation_amount']

        return {
            'total_violations': self.total_line_violations,
            'max_violation': self.max_line_violation,
            'line_violation_counts': line_violation_counts,
            'max_violation_by_line': max_violation_by_line,
            'total_violation_by_line': total_violation_by_line
        }

    def get_obs(self) -> dict:
        return self._get_observation()

    def export_timeseries_csv(self, save_path: str):
        import pandas as pd

        if len(self.operation_history) == 0:
            raise ValueError("operation_history is empty. Run an episode first.")

        df = pd.DataFrame(self.operation_history).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        if hasattr(self, "data") and len(self.data) >= len(df):
            df["pv_raw"] = self.data["solar_power"].iloc[:len(df)].to_numpy()
            df["wind_raw"] = self.data["wind_power"].iloc[:len(df)].to_numpy()

        df.to_csv(save_path, index=True, encoding="utf-8-sig")