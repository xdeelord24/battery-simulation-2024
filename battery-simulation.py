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
                 num_cells_series=13):
        """
        Initialize the battery simulator.

        :param capacity_ah: Initial battery capacity in Ah.
        :param charge_rate_a: Charging rate in A (nominal).
        :param discharge_rate_a: Discharge rate in A (nominal).
        :param charge_efficiency: Charge efficiency (0-1).
        :param discharge_efficiency: Discharge efficiency (0-1).
        :param degradation_rate: Rate of capacity degradation per full cycle.
        :param num_cells_series: Number of Li-ion cells in series.
        
        For a Li-ion cell:
        Full charge voltage ~4.2 V, fully discharged ~3.0 V.
        For num_cells_series=13, pack voltage ~54.6 V (full) to ~39.0 V (empty).
        """
        self.initial_capacity = capacity_ah
        self.capacity = capacity_ah
        self.charge_rate = charge_rate_a
        self.discharge_rate = discharge_rate_a
        self.charge_eff = charge_efficiency
        self.discharge_eff = discharge_efficiency
        self.degradation_rate = degradation_rate
        
        self.num_cells_series = num_cells_series
        self.v_min_cell = 3.0   # minimum voltage per cell at 0% SoC
        self.v_max_cell = 4.2   # maximum voltage per cell at 100% SoC

        # State of Charge in percentage
        self.soc = 100
        # Track the number of completed full cycles
        self.cycles = 0
        # Time steps
        self.time = 0
        
        # Flag to detect full cycles (0%->100%->0% or 100%->0%->100%)
        self.last_full_charge_reached = True
        
        self.history = {'time': [], 'SoC': [], 'Capacity': [], 'Cycles': [], 'Voltage': []}
        
        # Log initial state
        self.log_state()

    def get_voltage_from_soc(self, soc):
        """
        Estimate pack voltage from SoC using a simple linear approximation.
        Vpack(SOC) = Vmin_pack + (SOC/100)*(Vmax_pack - Vmin_pack)
        where:
        Vmax_pack = num_cells_series * v_max_cell
        Vmin_pack = num_cells_series * v_min_cell
        """
        v_min_pack = self.num_cells_series * self.v_min_cell
        v_max_pack = self.num_cells_series * self.v_max_cell
        voltage = v_min_pack + (soc / 100.0) * (v_max_pack - v_min_pack)
        return voltage

    def log_state(self):
        """Logs current state to history."""
        self.history['time'].append(self.time)
        self.history['SoC'].append(self.soc)
        self.history['Capacity'].append(self.capacity)
        self.history['Cycles'].append(self.cycles)
        # Calculate and log voltage
        self.history['Voltage'].append(self.get_voltage_from_soc(self.soc))

    def effective_charge_rate(self):
        """
        Returns an effective charge rate based on SoC.
        Reduce charge rate linearly above 80% SoC.
        """
        if self.soc >= 80:
            reduction_factor = 1.0 - 0.9 * ((self.soc - 80) / 20)
            return self.charge_rate * reduction_factor
        else:
            return self.charge_rate

    def effective_discharge_rate(self):
        """
        Returns an effective discharge rate based on SoC.
        Reduce discharge rate linearly below 20% SoC.
        """
        if self.soc <= 20:
            reduction_factor = 1.0 - 0.9 * ((20 - self.soc) / 20)
            return self.discharge_rate * reduction_factor
        else:
            return self.discharge_rate

    def charge(self, time_hours):
        """Simulates charging the battery for a given number of hours."""
        eff_rate = self.effective_charge_rate()
        energy_added = eff_rate * time_hours * self.charge_eff  # Ah added
        new_soc = self.soc + (energy_added / self.capacity * 100)
        if new_soc > 100:
            new_soc = 100
        self.update_soc(new_soc)

        if self.soc == 100:
            self.last_full_charge_reached = True

    def discharge(self, time_hours):
        """Simulates discharging the battery for a given number of hours."""
        eff_rate = self.effective_discharge_rate()
        energy_removed = eff_rate * time_hours / self.discharge_eff  # Ah removed
        new_soc = self.soc - (energy_removed / self.capacity * 100)
        if new_soc < 0:
            new_soc = 0
        old_soc = self.soc
        self.update_soc(new_soc)

        # If we reach 0% and we had previously been fully charged, that completes a full cycle
        if self.soc == 0 and self.last_full_charge_reached:
            self.cycles += 1
            self.last_full_charge_reached = False
            # Apply capacity degradation per full cycle
            self.capacity -= self.degradation_rate * self.initial_capacity
            if self.capacity < 0:
                self.capacity = 0

    def update_soc(self, new_soc):
        """
        Updates the State of Charge and time, and logs the state.
        """
        self.soc = max(min(new_soc, 100), 0)
        self.time += 1
        self.log_state()

    def simulate(self, steps, charge_time=1.0, discharge_time=1.0):
        """
        Simulates a series of charge/discharge cycles.
        
        :param steps: Number of steps.
        :param charge_time: Duration of each charging period (hours).
        :param discharge_time: Duration of each discharging period (hours).
        """
        for _ in range(steps):
            self.charge(charge_time)
            self.discharge(discharge_time)

    def plot_results(self):
        """Plots the State of Charge, Capacity, and Voltage over time."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # Plot SoC
        axs[0].plot(self.history['time'], self.history['SoC'], color='tab:blue')
        axs[0].set_ylabel('SoC (%)')
        axs[0].grid(True)

        # Plot Capacity
        axs[1].plot(self.history['time'], self.history['Capacity'], color='tab:red')
        axs[1].set_ylabel('Capacity (Ah)')
        axs[1].grid(True)

        # Plot Voltage
        axs[2].plot(self.history['time'], self.history['Voltage'], color='tab:green')
        axs[2].set_ylabel('Voltage (V)')
        axs[2].set_xlabel('Time Steps')
        axs[2].grid(True)

        plt.suptitle('Battery Simulation (SoC, Capacity, Voltage)')
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Realistic battery scenario:
    #  - Initial capacity = 100 Ah
    #  - Charge/Discharge rate = 10 A nominal
    #  - Charge & discharge efficiency = 95%
    #  - Slight capacity degradation per cycle
    #  - 13 cells in series, giving a voltage range from ~39 V (empty) to ~54.6 V (full)
    battery = BatterySimulator(capacity_ah=100, 
                               charge_rate_a=10, 
                               discharge_rate_a=10, 
                               charge_efficiency=0.95, 
                               discharge_efficiency=0.95, 
                               degradation_rate=0.0005,
                               num_cells_series=13)
    
    # Simulate 500 steps, each step: 0.5h charge, then 0.5h discharge
    battery.simulate(steps=10000, charge_time=0.5, discharge_time=0.5)
    
    # Plot results (SoC, Capacity, and Voltage)
    battery.plot_results()
