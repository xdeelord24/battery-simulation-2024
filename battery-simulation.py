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
                 thermal_cooling_rate=0.1,
                 thermal_heating_factor=0.05):

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

    def discharge(self, time_hours, discharge_rate_fraction=1.0):
        """Simulates battery discharging with a certain fraction of the max discharge rate."""
        eff_rate = self.effective_discharge_rate() * discharge_rate_fraction
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
        Very simplified thermal model.
        """
        total_current = charge_current + discharge_current
        heat_generated = self.thermal_heating_factor * total_current * time_hours
        self.temperature += heat_generated

        # Move temperature towards ambient
        temp_diff = self.temperature - self.ambient_temperature
        if abs(temp_diff) > 0.01:
            self.temperature -= np.sign(temp_diff) * min(abs(temp_diff), self.thermal_cooling_rate)

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

        :param night_charge_hours: Hours of continuous charging at night.
        :param day_hours: Total daytime hours available for usage events.
        :param usage_events: Number of random usage events (discharge periods).
        :param max_usage_duration: Maximum length (hours) of a single usage event.
        """
        # Night Charging: Long, steady charge
        # Charge in increments of 0.5 hours to simulate time steps
        charge_step = 0.5
        steps = int(night_charge_hours / charge_step)
        for _ in range(steps):
            self.charge(charge_step)

        # Daytime usage:
        # We have a total of `day_hours` to place `usage_events`.
        # We'll generate random start times and durations for each event.
        # Sort them by start time to simulate a sequence.

        events = []
        for _ in range(usage_events):
            event_start = random.uniform(0, day_hours - 0.1)
            event_duration = random.uniform(0.1, max_usage_duration)
            # Each event has a start and duration
            events.append((event_start, event_duration))

        # Sort events by start time
        events.sort(key=lambda x: x[0])

        current_time = 0.0
        timestep = 0.5  # half-hour increments for simulation

        for event_start, event_duration in events:
            # Idle until event_start
            while current_time < event_start:
                # No charging or discharging, just time passing
                self.idle(timestep)
                current_time += timestep

            # Event (discharge)
            event_end = event_start + event_duration
            while current_time < event_end:
                # Discharge at some fraction of the discharge rate
                # Let's assume random usage intensity: 30% to 100% of max discharge
                discharge_fraction = random.uniform(0.3, 1.0)
                self.discharge(timestep, discharge_rate_fraction=discharge_fraction)
                current_time += timestep

        # After last event, if any time remains in the day, remain idle
        while current_time < day_hours:
            self.idle(timestep)
            current_time += timestep

    def idle(self, time_hours):
        """Simulate idle time (no charging or discharging)."""
        # Just let temperature approach ambient
        self.heat_dissipation(charge_current=0, discharge_current=0, time_hours=time_hours)
        # Advance time steps
        steps = int(time_hours / 0.5)
        for _ in range(steps):
            # Logging at each half-hour for consistency
            self.time += 1
            self.log_state()

    def simulate(self, days=10):
        """
        Simulate multiple days of human-like usage:
        Each day:
          - 8 hours night charge
          - 16 hours day usage with random events

        :param days: Number of days to simulate.
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
    # Create a battery simulator and run for 10 days
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
        thermal_cooling_rate=0.05,
        thermal_heating_factor=0.05
    )

    # Simulate 10 days of human-like usage
    battery.simulate(days=10)

    # Plot the results
    battery.plot_results()
