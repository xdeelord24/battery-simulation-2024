import matplotlib.pyplot as plt
import numpy as np
import random

class BatterySimulator:
    def __init__(self, 
                 capacity_ah, 
                 charge_rate_a, 
                 discharge_rate_a, 
                 charge_efficiency=0.95, 
                 discharge_efficiency=0.95, 
                 degradation_rate=0.0001, 
                 num_cells_series=13,
                 initial_temperature=25.0,
                 ambient_temperature=25.0,
                 # Removed old thermal parameters in favor of a more physical model:
                 thermal_mass=5000.0,    # Joules/°C, thermal mass of the battery
                 internal_resistance=0.01,  # ohms, internal resistance for heat generation
                 cooling_coefficient=0.1,   # 1/hour, Newtonian cooling coefficient
                 time_step=0.5):  # hours per simulation step

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

        # Temperature and thermal model parameters
        self.temperature = initial_temperature
        self.ambient_temperature = ambient_temperature
        self.thermal_mass = thermal_mass
        self.internal_resistance = internal_resistance
        self.cooling_coefficient = cooling_coefficient

        self.time_step = time_step

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
        """Reduce charge rate near full capacity."""
        if self.soc >= 80:
            reduction_factor = 1.0 - 0.9 * ((self.soc - 80) / 20)
            return self.charge_rate * reduction_factor
        else:
            return self.charge_rate

    def effective_discharge_rate(self):
        """Reduce discharge rate near empty capacity."""
        if self.soc <= 20:
            reduction_factor = 1.0 - 0.9 * ((20 - self.soc) / 20)
            return self.discharge_rate * reduction_factor
        else:
            return self.discharge_rate

    def temperature_adjusted_efficiencies(self):
        """
        Adjust charge/discharge efficiencies and capacity based on temperature.
        """
        charge_eff = self.charge_eff_base
        discharge_eff = self.discharge_eff_base
        capacity_factor = 1.0

        if self.temperature < 10:
            charge_eff *= 0.9
            discharge_eff *= 0.9
            capacity_factor = 0.95
        elif self.temperature > 40:
            charge_eff *= 0.9
            discharge_eff *= 0.9
            capacity_factor = 0.95

        return charge_eff, discharge_eff, capacity_factor

    def charge(self, time_hours):
        """Simulates battery charging."""
        eff_rate = self.effective_charge_rate()
        charge_eff, _, capacity_factor = self.temperature_adjusted_efficiencies()

        effective_capacity = self.capacity * capacity_factor
        energy_added = eff_rate * time_hours * charge_eff
        new_soc = self.soc + (energy_added / effective_capacity * 100)
        new_soc = min(new_soc, 100)
        self.update_soc(new_soc)

        # Heating effect from charging
        self.heat_dissipation(charge_current=eff_rate, discharge_current=0, time_hours=time_hours)

        if self.soc == 100:
            self.last_full_charge_reached = True

    def discharge(self, time_hours, discharge_rate_fraction=1.0):
        """Simulates battery discharging with a certain fraction of the max discharge rate."""
        eff_rate = self.effective_discharge_rate() * discharge_rate_fraction
        _, discharge_eff, capacity_factor = self.temperature_adjusted_efficiencies()

        effective_capacity = self.capacity * capacity_factor
        energy_removed = eff_rate * time_hours / discharge_eff
        new_soc = self.soc - (energy_removed / effective_capacity * 100)
        new_soc = max(new_soc, 0)
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
        Improved thermal model:
        - Compute heat from I²R losses
        - Use thermal mass to determine temperature rise
        - Apply Newtonian cooling to ambient
        """
        total_current = charge_current + discharge_current

        # Compute I²R losses over the given time period:
        # Power (W) = I²R; Energy (J) = Power * time(s)
        energy_lost_joules = (total_current**2 * self.internal_resistance) * (time_hours * 3600)
        
        # Convert energy to temperature rise based on thermal mass
        delta_temp = energy_lost_joules / self.thermal_mass
        self.temperature += delta_temp

        # Apply Newtonian cooling
        temp_diff = self.temperature - self.ambient_temperature
        # dT = -k * (T - Tamb) * time_hours
        self.temperature -= self.cooling_coefficient * temp_diff * time_hours

    def update_soc(self, new_soc):
        """Updates SOC, time, and logs the state."""
        self.soc = max(min(new_soc, 100), 0)
        self.time += 1
        self.log_state()

    def log_state(self):
        """Logs current state."""
        self.history['time'].append(self.time)
        self.history['SoC'].append(self.soc)
        self.history['Capacity'].append(self.capacity)
        self.history['Cycles'].append(self.cycles)
        self.history['Voltage'].append(self.get_voltage_from_soc(self.soc))
        self.history['Temperature'].append(self.temperature)

    def simulate_day(self, 
                     night_charge_hours=8.0, 
                     day_hours=16.0, 
                     usage_events=5, 
                     max_usage_duration=1.0):
        """
        Simulate one "day":
        - Night: charge for `night_charge_hours` at a steady rate.
        - Day: random usage events (discharges) scattered throughout `day_hours`.
        """
        # Night Charging: 
        steps = int(night_charge_hours / self.time_step)
        for _ in range(steps):
            self.charge(self.time_step)

        # Daytime usage:
        events = []
        for _ in range(usage_events):
            event_start = random.uniform(0, day_hours - 0.1)
            event_duration = random.uniform(0.1, max_usage_duration)
            events.append((event_start, event_duration))

        # Sort events by start time
        events.sort(key=lambda x: x[0])

        current_time = 0.0
        while events:
            event_start, event_duration = events.pop(0)

            # Idle until event_start
            while current_time < event_start:
                self.idle(min(self.time_step, event_start - current_time))
                current_time += self.time_step

            # Event (discharge)
            event_end = event_start + event_duration
            while current_time < event_end:
                discharge_fraction = random.uniform(0.3, 1.0)
                self.discharge(self.time_step, discharge_rate_fraction=discharge_fraction)
                current_time += self.time_step

        # After last event, if any time remains in the day, remain idle
        while current_time < day_hours:
            self.idle(min(self.time_step, day_hours - current_time))
            current_time += self.time_step

    def idle(self, time_hours):
        """Simulate idle time (no charging or discharging)."""
        # Even when idle, run thermal model (just cooling)
        # When no current: no added heat, just cooling step.
        # But for consistency, let's apply a small step for temperature each time_step.
        steps = int(time_hours / self.time_step)
        remainder = time_hours - steps * self.time_step
        for _ in range(steps):
            self.heat_dissipation(charge_current=0, discharge_current=0, time_hours=self.time_step)
            self.time += 1
            self.log_state()
        if remainder > 0:
            self.heat_dissipation(charge_current=0, discharge_current=0, time_hours=remainder)
            self.time += 1
            self.log_state()

    def simulate(self, days=10):
        """
        Simulate multiple days of human-like usage.
        Each day:
          - 8 hours night charge
          - 16 hours day usage with random events
        """
        for day in range(days):
            self.simulate_day(night_charge_hours=8, 
                              day_hours=16, 
                              usage_events=random.randint(3,7), 
                              max_usage_duration=2.0)

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
        axs[3].set_ylabel('Temp (°C)')
        axs[3].set_xlabel('Time Steps')
        axs[3].grid(True)

        plt.suptitle('Human-like Battery Usage Over Multiple Days')
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
        thermal_mass=5000.0,       # Joules/°C
        internal_resistance=0.01,  # Ohms
        cooling_coefficient=0.1,   # 1/hour
        time_step=0.5              # hours per step
    )

    # Simulate 10 days of human-like usage
    battery.simulate(days=10)

    # Plot the results
    battery.plot_results()
