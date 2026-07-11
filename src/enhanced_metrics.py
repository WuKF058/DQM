import pandas as pd
import numpy as np


class EnhancedMetrics:

    @staticmethod
    def calculate_economic_metrics(operation_history, data, config):
        df = pd.DataFrame(operation_history)

        total_operating_cost = df['cumulative_cost'].iloc[-1] if 'cumulative_cost' in df.columns else 0

        diesel_energy = df['diesel_power'].sum() / 4  # kWh
        grid_buy_energy = df[df['grid_power'] > 0]['grid_power'].sum() / 4  # kWh
        grid_sell_energy = df[df['grid_power'] < 0]['grid_power'].sum() / 4  # kWh

        diesel_cost = diesel_energy * config.get("diesel_cost", 0.15)
        grid_buy_cost = grid_buy_energy * config.get("grid_buy_price", 0.12)
        grid_sell_income = abs(grid_sell_energy) * config.get("grid_sell_price", 0.08)

        line_violation_penalty = 0
        if 'has_line_violation' in df.columns:
            avg_violation = df[df['has_line_violation'] == True].shape[0] / len(df) if len(df) > 0 else 0
            line_violation_penalty = avg_violation * config.get("line_violation_penalty_coef",
                                                                15.0) * 100  # 假设平均越限100kW

        metrics = {
            'total_operating_cost': total_operating_cost,
            'diesel_cost': diesel_cost,
            'grid_buy_cost': grid_buy_cost,
            'grid_sell_income': grid_sell_income,
            'line_violation_penalty': line_violation_penalty,
            'net_energy_cost': diesel_cost + grid_buy_cost - grid_sell_income + line_violation_penalty,
            'average_daily_cost': total_operating_cost / (len(df) / 96),  # 假设每天96个15分钟间隔
        }
        return metrics

    @staticmethod
    def calculate_reliability_metrics(operation_history):
        df = pd.DataFrame(operation_history)

        total_load_shedding = df['load_shedding'].sum() / 4  # kWh
        total_load_demand = df['load_demand'].sum() / 4 if 'load_demand' in df.columns else 1

        line_violation_frequency = 0
        if 'has_line_violation' in df.columns:
            line_violation_frequency = (df['has_line_violation'] == True).sum() / len(df)

        return {
            'load_shedding_frequency': (df['load_shedding'] > 0).sum() / len(df),
            'average_load_shedding': df['load_shedding'].mean(),
            'total_load_shedding_kwh': total_load_shedding,
            'reliability_index': 1 - (total_load_shedding / total_load_demand) if total_load_demand > 0 else 1,
            'supply_availability': 1 - (total_load_shedding / total_load_demand) if total_load_demand > 0 else 1,
            'line_violation_frequency': line_violation_frequency,
            'line_constraint_reliability': 1 - line_violation_frequency
        }

    @staticmethod
    def calculate_environmental_metrics(operation_history, data):
        df = pd.DataFrame(operation_history)

        total_solar = data['solar_power'].sum() / 4  # kWh
        total_wind = data['wind_power'].sum() / 4  # kWh
        total_renewable = total_solar + total_wind

        total_curtailment = df['renewable_curtailment'].sum() / 4  # kWh

        diesel_emissions = df['diesel_power'].sum() / 4 * 0.8

        return {
            'carbon_emissions_kg': diesel_emissions,
            'renewable_utilization_rate': (
                                                  total_renewable - total_curtailment) / total_renewable if total_renewable > 0 else 0,
            'clean_energy_ratio': (total_renewable - total_curtailment) / (
                    total_renewable + df['diesel_power'].sum() / 4) if (total_renewable + df[
                'diesel_power'].sum() / 4) > 0 else 0,
            'total_renewable_energy_kwh': total_renewable,
            'renewable_curtailment_kwh': total_curtailment,
            'renewable_penetration_rate': total_renewable / (total_renewable + df['diesel_power'].sum() / 4) if (
                                                                                                                        total_renewable +
                                                                                                                        df[
                                                                                                                            'diesel_power'].sum() / 4) > 0 else 0
        }

    @staticmethod
    def calculate_line_constraint_metrics(operation_history):
        df = pd.DataFrame(operation_history)

        if 'has_line_violation' not in df.columns:
            return {}

        violation_df = df[df['has_line_violation'] == True]
        total_violations = len(violation_df)
        violation_frequency = total_violations / len(df) if len(df) > 0 else 0

        line_violation_details = {}
        if 'line_violations' in df.columns and len(violation_df) > 0:
            line_counts = {}
            for violations in violation_df['line_violations']:
                if isinstance(violations, dict):
                    for line_name in violations.keys():
                        line_counts[line_name] = line_counts.get(line_name, 0) + 1

            line_violation_details = {
                'total_violations': total_violations,
                'violation_frequency': violation_frequency,
                'line_violation_counts': line_counts,
                'average_violations_per_step': total_violations / len(df) if len(df) > 0 else 0,
                'max_consecutive_violations': EnhancedMetrics._calculate_max_consecutive_violations(df)
            }

        return line_violation_details

    @staticmethod
    def _calculate_max_consecutive_violations(df):
        max_consecutive = 0
        current_consecutive = 0

        for has_violation in df.get('has_line_violation', []):
            if has_violation:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive