import os
import numpy as np
import torch

class QuantumConfig:
    # Quantum Circuit Parameters
    n_qubits = 4               # Number of Qubit
    n_layers = 2                # Number of Quantum Layer
    ansatz_type = 1             # Type of Quantum Circuit
                                # 1 = Fast Hardware,
                                # 2=CRY Gate
    shots = 1024                # Shots

    entropy_coef = 0.01  # Entropy
    use_parameter_shift = True  # if turn on the Parameter_Shift
    visualize_circuit = True  # if circuit is visual

    # Noise
    depolarising_error = False
    gate_control_noise = False
    noise_level = 0.01          # The strength of Noise

    # Optimize Parameters
    use_parameter_shift = True
    learning_rate = 0.0005       # Learning rate
    gamma = 0.99
    entropy_coef = 0.01

    # The limit of the available resource
    max_quantum_params = 64
    min_qubits = 1
    max_qubits = 8

    log_quantum_metrics = True

    compare_classical = False
    use_parameter_shift = True
    visualize_circuit = Tru

    # Setting of Device
    @property
    def device(self):
        """Choose the Training Device"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def quantum_device(self):
        """
        Get the Quantum Device.
        You can use qiskit-machine-learning or pennylane
        """
        import pennylane as qml
        if self.depolarising_error:
            return qml.device("default.mixed", wires=self.n_qubits)
        return qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)

    def get_circuit_params(self):
        """Get the parameters of the circuit."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "ansatz": self.ansatz_type,
            "depolarising_error": self.depolarising_error,
            "gate_control_noise": self.gate_control_noise
        }


# config.py (更新部分)
class MicrogridConfig:
    """
    The hyperparameters of Microgrids
    """
    pv_capacity = 100  # kW
    wind_capacity = 50  # kW
    diesel_capacity = 200  # kW
    battery_capacity = 500  # kWh

    line_capacity_pv = 120  # kW
    line_capacity_wind = 60  # kW
    line_capacity_diesel = 250  # kW
    line_capacity_battery = 300  # kW
    line_capacity_grid = 400  # kW
    line_capacity_load = 500  # kW

    line_safety_margin = 0.9

    battery_soc_min = 0.2
    battery_soc_max = 0.95
    battery_efficiency = 0.95
    battery_max_power_ratio = 0.6
    battery_soc_penalty_high = 0.85
    battery_soc_penalty_low = 0.25
    diesel_fuel_capacity = 3  # kWh
    fuel_consumption_rate = 0.1

    diesel_cost = 0.15  # €/kWh
    grid_buy_price = 0.12  # €/kWh
    grid_sell_price = 0.08  # €/kWh
    curtailment_penalty = 0.5  # €/kWh
    load_shedding_penalty = 2.0  # €/kWh

    dynamic_curtailment_penalty = True
    load_demand_scale = 3.0

    @property
    def device(self):
        """Auto-Choose the Training Device"""
        return "cuda" if torch.cuda.is_available() else "cpu"

class TrainingConfig:
    batch_size = 64
    buffer_size = 10000
    update_frequency = 100

    learning_rate = 0.0005
    weight_decay = 0.0001
    momentum = 0.9

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    log_interval = 10
    save_interval = 100
    checkpoint_dir = "checkpoints"


    def __init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class A2CConfig:
    gamma = 0.99
    entropy_beta = 0.01
    value_loss_coef = 0.5
    policy_loss_coef = 1.0
    max_grad_norm = 0.5

    lstm_hidden_size = 256
    lstm_num_layers = 1

    net_arch = {
        "pi": [128, 128],
        "vf": []
    }
    actor_hidden_sizes = [128, 128]
    critic_hidden_sizes = [128, 64]

    num_epochs = 10
    num_minibatches = 4
    clip_param = 0.2


class NavQConfig(QuantumConfig, A2CConfig):
    def __init__(self):
        self.learning_rate = QuantumConfig.learning_rate

        if self.n_qubits * self.n_layers * 2 > self.max_quantum_params:
            raise ValueError(f"The number of Quantum Parameters ({self.n_qubits*self.n_layers*2}) Exceeded the limit {self.max_quantum_params}")

        # 创建必要的目录
        os.makedirs("quantum_circuits", exist_ok=True)

    @property
    def hybrid_params(self):
        return {
            "quantum": self.get_circuit_params(),
            "classical": {
                "actor_hidden_sizes": self.actor_hidden_sizes,
                "critic_hidden_sizes": self.critic_hidden_sizes,
                "lstm_hidden_size": self.lstm_hidden_size
            }
        }

    def get_wandb_config(self):
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "ansatz_type": self.ansatz_type,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef
        }


class CARLAConfig:
    host = "localhost"
    port = 2000
    timeout = 10.0

    render_width = 1280
    render_height = 720
    render_fps = 20

    vehicle_model = "audi.tt"
    max_steering_angle = 70
    max_throttle = 1.0
    max_brake = 1.0

    lidar_channels = 32
    camera_fov = 90
    semantic_segmentation = True


class Config:
    def __init__(self):
        self.quantum = QuantumConfig()
        self.microgrid = MicrogridConfig()
        self.training = TrainingConfig()
        self.a2c = A2CConfig()
        self.navq = NavQConfig()
        self.carla = CARLAConfig()

        self._compute_dependent_params()

    def _compute_dependent_params(self):
        self.navq.critic_input_dim = 3 * self.quantum.n_qubits
        self.microgrid.state_dim = 8  # 微电网状态维度

        self.quantum.total_params = (
                self.quantum.n_layers *
                (self.quantum.n_qubits * 2) +
                self.quantum.n_qubits
        )

config = Config()