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
                 thermal_mass=5000.0,    # Joules/°C
                 internal_resistance=0.01,  # ohms
                 cooling_coefficient=0.1,   # 1/hour
                 time_step=0.5,             # hours per simulation step
                 human_behavior=True,
                 max_depth_of_discharge=80.0,  # % of full capacity that can be used (e.g., 80% means don't go below 20% SoC)
                 max_temperature=40.0,          # °C, try not to exceed this temperature
                 gentle_charge_rate_factor=0.5  # Reduce charge rate by this factor if conditions demand gentler charging
                 ):
        
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

        # Thermal model parameters
        self.temperature = initial_temperature
        self.ambient_temperature = ambient_temperature
        self.thermal_mass = thermal_mass
        self.internal_resistance = internal_resistance
        self.cooling_coefficient = cooling_coefficient

        self.time_step = time_step
        self.human_behavior = human_behavior

        # Additional best-practice parameters
        self.max_depth_of_discharge = max_depth_of_discharge
        self.min_soc = 100 - self.max_depth_of_discharge  # For 80% DoD, min SoC is 20%
        self.max_temperature = max_temperature
        self.gentle_charge_rate_factor = gentle_charge_rate_factor

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
        """Calculate effective charge rate considering SoC, temperature, and gentle charge factor."""
        # Base effective charge rate (reducing near full)
        if self.soc >= 80:
            reduction_factor = 1.0 - 0.9 * ((self.soc - 80) / 20)
            base_rate = self.charge_rate * reduction_factor
        else:
            base_rate = self.charge_rate

        # If temperature is high or we want gentler charging near high SoC, reduce further
        if self.temperature > self.max_temperature or self.soc > 90:
            base_rate *= self.gentle_charge_rate_factor

        return base_rate

    def effective_discharge_rate(self):
        """Calculate effective discharge rate considering SoC and temperature limits."""
        # Avoid going below min_soc for longevity
        if self.soc <= self.min_soc:
            # If near minimum SoC, drastically reduce discharge to prevent going below
            reduction_factor = max(0.0, (self.soc - self.min_soc) / (20.0))  # a simple scale
            base_rate = self.discharge_rate * reduction_factor
        else:
            base_rate = self.discharge_rate

        # If temperature is too high, reduce discharge to avoid further heating
        if self.temperature > self.max_temperature:
            base_rate *= 0.5

        return base_rate

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
        """Simulates battery charging respecting gentle charging under certain conditions."""
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
        """Simulates battery discharging within allowed DOD and temperature constraints."""
        eff_rate = self.effective_discharge_rate() * discharge_rate_fraction
        if eff_rate < 0.01:
            # Too low to effectively discharge due to constraints, just idle
            self.idle(time_hours)
            return

        _, discharge_eff, capacity_factor = self.temperature_adjusted_efficiencies()

        effective_capacity = self.capacity * capacity_factor
        energy_removed = eff_rate * time_hours / discharge_eff
        new_soc = self.soc - (energy_removed / effective_capacity * 100)
        # Prevent going below the allowed minimum SoC
        new_soc = max(new_soc, self.min_soc)
        self.update_soc(new_soc)

        # Heating effect from discharging
        self.heat_dissipation(charge_current=0, discharge_current=eff_rate, time_hours=time_hours)

        # Full cycle detection (only if min_soc is effectively 0, else partial cycles)
        # For partial cycles, you may define cycle counting differently.
        if self.min_soc <= 0 and self.soc == 0 and self.last_full_charge_reached:
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

        # Compute I²R losses:
        # Power (W) = I²R; Energy (J) = Power * time(s)
        energy_lost_joules = (total_current**2 * self.internal_resistance) * (time_hours * 3600)
        
        # Convert energy to temperature rise based on thermal mass
        delta_temp = energy_lost_joules / self.thermal_mass
        self.temperature += delta_temp

        # Apply Newtonian cooling
        temp_diff = self.temperature - self.ambient_temperature
        self.temperature -= self.cooling_coefficient * temp_diff * time_hours

    def check_and_resync(self):
        """
        Checks if the battery SoC needs resynchronization based on voltage.
        """
        current_voltage = self.get_voltage_from_soc(self.soc)
        expected_full_voltage = self.num_cells_series * self.v_max_cell
        expected_empty_voltage = self.num_cells_series * self.v_min_cell

        tolerance = 0.01

        # If at or near full voltage but SOC isn't 100%, resync
        if abs(current_voltage - expected_full_voltage) < tolerance and self.soc < 99.9:
            self.soc = 100.0

        # If at or near empty voltage but SOC isn't min_soc, resync
        # (Only if min_soc == 0, otherwise min_soc is your chosen lower limit)
        if self.min_soc == 0 and abs(current_voltage - expected_empty_voltage) < tolerance and self.soc > 0.1:
            self.soc = 0.0

    def update_soc(self, new_soc):
        """Updates SOC, time, and logs the state."""
        self.soc = max(min(new_soc, 100), self.min_soc)
        self.check_and_resync()
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
        - Day: If human_behavior=True, random usage events (discharges).
               If human_behavior=False, simple continuous discharge pattern.
        """
        # Night Charging:
        steps = int(night_charge_hours / self.time_step)
        for _ in range(steps):
            self.charge(self.time_step)

        if self.human_behavior:
            # Daytime usage with random events
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
        else:
            # If not human behavior, just continuously discharge over the day at a constant rate
            current_time = 0.0
            steps = int(day_hours / self.time_step)
            for _ in range(steps):
                self.discharge(self.time_step, discharge_rate_fraction=0.5)
                current_time += self.time_step

    def idle(self, time_hours):
        """Simulate idle time (no charging or discharging)."""
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
        Simulate multiple days of usage.
        If human_behavior=True, usage pattern is randomized events.
        If human_behavior=False, usage pattern is simple continuous discharge.

        The code now also implements best practices:
        - Limited DoD (min_soc) to reduce stress
        - Lower charge and discharge rates at high temp or high SoC
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

        plt.suptitle('Battery Usage With Best-Practice Constraints')
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Human-like usage with additional best practice constraints
    battery_human = BatterySimulator(
        capacity_ah=100,
        charge_rate_a=10,
        discharge_rate_a=10,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        degradation_rate=0.0005,
        num_cells_series=13,
        initial_temperature=25.0,
        ambient_temperature=25.0,
        thermal_mass=5000.0,
        internal_resistance=0.01,
        cooling_coefficient=0.1,
        time_step=0.5,
        human_behavior=True,
        max_depth_of_discharge=80.0,
        max_temperature=40.0,
        gentle_charge_rate_factor=0.5
    )
    battery_human.simulate(days=10)
    battery_human.plot_results()

    # Non-human usage scenario with best practice constraints
    battery_simple = BatterySimulator(
        capacity_ah=100,
        charge_rate_a=10,
        discharge_rate_a=10,
        charge_efficiency=0.95,
        discharge_efficiency=0.95,
        degradation_rate=0.0005,
        num_cells_series=13,
        initial_temperature=25.0,
        ambient_temperature=25.0,
        thermal_mass=5000.0,
        internal_resistance=0.01,
        cooling_coefficient=0.1,
        time_step=0.5,
        human_behavior=False,
        max_depth_of_discharge=80.0,
        max_temperature=40.0,
        gentle_charge_rate_factor=0.5
    )
    battery_simple.simulate(days=10)
    battery_simple.plot_results()
