import matplotlib.pyplot as plt
import numpy as np

class BatterySimulator:
    def __init__(self, capacity_ah, charge_rate_a, discharge_rate_a, efficiency=0.95, degradation_rate=0.0001):
        """
        Initialize the battery simulator.

        :param capacity_ah: Initial battery capacity in Ah.
        :param charge_rate_a: Charging rate in A.
        :param discharge_rate_a: Discharge rate in A.
        :param efficiency: Charge/discharge efficiency (0-1).
        :param degradation_rate: Rate of capacity degradation per cycle.
        """
        self.initial_capacity = capacity_ah
        self.capacity = capacity_ah
        self.charge_rate = charge_rate_a
        self.discharge_rate = discharge_rate_a
        self.efficiency = efficiency
        self.degradation_rate = degradation_rate
        self.soc = 100  # State of Charge in percentage
        self.cycles = 0
        self.time = 0
        self.history = {'time': [], 'SoC': [], 'Capacity': []}

    def charge(self, time_hours):
        """Simulates charging the battery."""
        energy_added = self.charge_rate * time_hours * self.efficiency  # Energy added in Ah
        new_soc = self.soc + (energy_added / self.capacity * 100)
        if new_soc > 100:
            new_soc = 100
        self.update_soc(new_soc)

    def discharge(self, time_hours):
        """Simulates discharging the battery."""
        energy_removed = self.discharge_rate * time_hours / self.efficiency  # Energy removed in Ah
        new_soc = self.soc - (energy_removed / self.capacity * 100)
        if new_soc < 0:
            new_soc = 0
        self.update_soc(new_soc)

    def update_soc(self, new_soc):
        """Updates the State of Charge and capacity based on degradation."""
        if self.soc > 0 and new_soc == 0:  # Full discharge detected, counts as a cycle
            self.cycles += 1
            self.capacity -= self.degradation_rate * self.initial_capacity  # Capacity degradation
        self.soc = max(min(new_soc, 100), 0)
        self.time += 1
        self.history['time'].append(self.time)
        self.history['SoC'].append(self.soc)
        self.history['Capacity'].append(self.capacity)

    def simulate(self, steps, charge_time=1, discharge_time=1):
        """Simulates a series of charge and discharge cycles."""
        for _ in range(steps):
            self.charge(charge_time)
            self.discharge(discharge_time)

    def plot_results(self):
        """Plots the State of Charge and Capacity over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['time'], self.history['SoC'], label='State of Charge (SoC)')
        plt.plot(self.history['time'], self.history['Capacity'], label='Battery Capacity (Ah)')
        plt.xlabel('Time Steps')
        plt.ylabel('Percentage / Capacity')
        plt.title('Battery Simulation')
        plt.legend()
        plt.grid()
        plt.show()


# # Simulation Example
if __name__ == "__main__":
    battery = BatterySimulator(capacity_ah=100, charge_rate_a=10, discharge_rate_a=10, efficiency=0.95, degradation_rate=0.0005)
    battery.simulate(steps=200, charge_time=0.5, discharge_time=0.5)
    battery.plot_results()

# if __name__ == "__main__":
#     # Simulating a battery with 100 Ah capacity, charging/discharging at 10 A
#     battery = BatterySimulator(capacity_ah=100, charge_rate_a=10, discharge_rate_a=10, 
#                                efficiency=0.95, degradation_rate=0.0005)
    
#     # Simulate 500 cycles: 0.5 hours charge + 0.5 hours discharge in each step
#     battery.simulate(steps=500, charge_time=5, discharge_time=5)
    
#     # Plot the results
#     battery.plot_results()
