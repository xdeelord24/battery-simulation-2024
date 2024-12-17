import matplotlib.pyplot as plt
import numpy as np

class BatterySimulator:
    def __init__(self, 
                 capacity_ah, 
                 charge_rate_a, 
                 discharge_rate_a, 
                 charge_efficiency=0.95, 
                 discharge_efficiency=0.95, 
                 degradation_rate=0.0001, 
                 num_cells_series=13,
                 initial_temperature=25.0,      # Starting temperature in °C
                 ambient_temperature=25.0,      # Ambient temperature in °C
                 thermal_cooling_rate=0.1,      # How quickly battery cools per time step towards ambient (°C per step)
                 thermal_heating_factor=0.05):  # How much temperature rises per A of current * hours
        """
        Initialize the battery simulator.

        :param capacity_ah: Initial battery capacity in Ah.
        :param charge_rate_a: Max charging rate (A).
        :param discharge_rate_a: Max discharging rate (A).
        :param charge_efficiency: Charge efficiency (0-1).
        :param discharge_efficiency: Discharge efficiency (0-1).
        :param degradation_rate: Rate of capacity degradation per full cycle.
        :param num_cells_series: Number of cells in series.
        :param initial_temperature: Initial temperature of the battery in °C.
        :param ambient_temperature: Ambient environmental temperature in °C.
        :param thermal_cooling_rate: Cooling rate towards ambient per time step.
        :param thermal_heating_factor: Factor for temperature rise due to current flow.
        """

        # Battery parameters
        self.initial_capacity = capacity_ah
        self.capacity = capacity_ah
        self.charge_rate = charge_rate_a
        self.discharge_rate = discharge_rate_a
        self.charge_eff_base = charge_efficiency
        self.discharge_eff_base = discharge_efficiency
        self.degradation_rate = degradation_rate
        self.num_cells_series = num_cells_series

        # Voltage parameters
        self.v_min_cell = 3.0   
        self.v_max_cell = 4.2   

        # State
        self.soc = 100
        self.cycles = 0
        self.time = 0

        # Temperature parameters
        self.temperature = initial_temperature
        self.ambient_temperature = ambient_temperature
        self.thermal_cooling_rate = thermal_cooling_rate
        self.thermal_heating_factor = thermal_heating_factor

        # Cycle tracking
        self.last_full_charge_reached = True

        # Data history
        self.history = {'time': [], 'SoC': [], 'Capacity': [], 'Cycles': [], 'Voltage': [], 'Temperature': []}

        self.log_state()

    def get_voltage_from_soc(self, soc):
        """Linear approximation of voltage from SoC."""
        v_min_pack = self.num_cells_series * self.v_min_cell
        v_max_pack = self.num_cells_series * self.v_max_cell
        voltage = v_min_pack + (soc / 100.0) * (v_max_pack - v_min_pack)
        return voltage

    def effective_charge_rate(self):
        """
        Reduce charge rate near full capacity.
        """
        if self.soc >= 80:
            reduction_factor = 1.0 - 0.9 * ((self.soc - 80) / 20)
            return self.charge_rate * reduction_factor
        else:
            return self.charge_rate

    def effective_discharge_rate(self):
        """
        Reduce discharge rate near empty capacity.
        """
        if self.soc <= 20:
            reduction_factor = 1.0 - 0.9 * ((20 - self.soc) / 20)
            return self.discharge_rate * reduction_factor
        else:
            return self.discharge_rate

    def temperature_adjusted_efficiencies(self):
        """
        Adjust charge and discharge efficiencies based on temperature.
        As a simplistic model:
        - At low temperatures (<10°C), efficiency decreases by up to 10%.
        - At high temperatures (>40°C), efficiency also decreases by up to 10%.
        Between 10°C and 40°C, the nominal efficiency is retained.

        Similarly, capacity can be affected by temperature:
        - At low temperature (<10°C), reduce usable capacity by 5%.
        - At high temperature (>40°C), reduce usable capacity by 5%.
        """
        charge_eff = self.charge_eff_base
        discharge_eff = self.discharge_eff_base
        capacity_factor = 1.0

        if self.temperature < 10:
            # Cold reduces efficiency and capacity
            charge_eff *= 0.9
            discharge_eff *= 0.9
            capacity_factor = 0.95
        elif self.temperature > 40:
            # Hot reduces efficiency and capacity
            charge_eff *= 0.9
            discharge_eff *= 0.9
            capacity_factor = 0.95

        return charge_eff, discharge_eff, capacity_factor

    def charge(self, time_hours):
        """Simulates battery charging."""
        eff_rate = self.effective_charge_rate()
        charge_eff, _, capacity_factor = self.temperature_adjusted_efficiencies()

        # Adjust the effective capacity based on temperature
        effective_capacity = self.capacity * capacity_factor

        energy_added = eff_rate * time_hours * charge_eff
        new_soc = self.soc + (energy_added / effective_capacity * 100)
        new_soc = min(new_soc, 100)
        self.update_soc(new_soc)

        # Heating effect from charging
        self.heat_dissipation(charge_current=eff_rate, discharge_current=0, time_hours=time_hours)

        if self.soc == 100:
            self.last_full_charge_reached = True

    def discharge(self, time_hours):
        """Simulates battery discharging."""
        eff_rate = self.effective_discharge_rate()
        _, discharge_eff, capacity_factor = self.temperature_adjusted_efficiencies()

        # Adjust the effective capacity based on temperature
        effective_capacity = self.capacity * capacity_factor

        energy_removed = eff_rate * time_hours / discharge_eff
        new_soc = self.soc - (energy_removed / effective_capacity * 100)
        new_soc = max(new_soc, 0)
        old_soc = self.soc
        self.update_soc(new_soc)

        # Heating effect from discharging
        self.heat_dissipation(charge_current=0, discharge_current=eff_rate, time_hours=time_hours)

        # Full cycle detection
        if self.soc == 0 and self.last_full_charge_reached:
            self.cycles += 1
            self.last_full_charge_reached = False
            # Degrade capacity after full cycle
            self.capacity -= self.degradation_rate * self.initial_capacity
            if self.capacity < 0:
                self.capacity = 0

    def heat_dissipation(self, charge_current, discharge_current, time_hours):
        """
        A simple temperature model:
        - Heat generated is proportional to current * time.
        - Battery then cools towards ambient at a fixed rate.
        """
        # Heating proportional to total current flow
        total_current = charge_current + discharge_current
        heat_generated = self.thermal_heating_factor * total_current * time_hours
        self.temperature += heat_generated

        # Cooling towards ambient
        # The closer the battery is to ambient, the less it cools.
        temp_diff = self.temperature - self.ambient_temperature
        if abs(temp_diff) > 0.01:
            # Move temperature a fraction closer to ambient
            self.temperature -= np.sign(temp_diff) * min(abs(temp_diff), self.thermal_cooling_rate)

    def update_soc(self, new_soc):
        """
        Updates SOC, time, and logs the state.
        """
        self.soc = max(min(new_soc, 100), 0)
        self.time += 1
        self.log_state()

    def log_state(self):
        """Logs current state to history."""
        self.history['time'].append(self.time)
        self.history['SoC'].append(self.soc)
        self.history['Capacity'].append(self.capacity)
        self.history['Cycles'].append(self.cycles)
        self.history['Voltage'].append(self.get_voltage_from_soc(self.soc))
        self.history['Temperature'].append(self.temperature)

    def simulate(self, steps, charge_time=1.0, discharge_time=1.0):
        """
        Simulates charge/discharge steps.
        """
        for _ in range(steps):
            self.charge(charge_time)
            self.discharge(discharge_time)

    def plot_results(self):
        """Plot SoC, Capacity, Voltage, and Temperature."""
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # SoC
        axs[0].plot(self.history['time'], self.history['SoC'], color='tab:blue')
        axs[0].set_ylabel('SoC (%)')
        axs[0].grid(True)

        # Capacity
        axs[1].plot(self.history['time'], self.history['Capacity'], color='tab:red')
        axs[1].set_ylabel('Capacity (Ah)')
        axs[1].grid(True)

        # Voltage
        axs[2].plot(self.history['time'], self.history['Voltage'], color='tab:green')
        axs[2].set_ylabel('Voltage (V)')
        axs[2].grid(True)

        # Temperature
        axs[3].plot(self.history['time'], self.history['Temperature'], color='tab:orange')
        axs[3].set_ylabel('Temperature (°C)')
        axs[3].set_xlabel('Time Steps')
        axs[3].grid(True)

        plt.suptitle('Battery Simulation (SoC, Capacity, Voltage, Temperature)')
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    battery = BatterySimulator(
        capacity_ah=100,
        charge_rate_a=10,
        discharge_rate_a=10,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        degradation_rate=0.0005,
        num_cells_series=13,
        initial_temperature=25.0,
        ambient_temperature=25.0,
        thermal_cooling_rate=0.05,    # slower cooling
        thermal_heating_factor=0.1    # more heat per current flow
    )

    # Simulate 100 steps with 0.5h charge and 0.5h discharge
    battery.simulate(steps=100, charge_time=0.5, discharge_time=0.5)
    battery.plot_results()
