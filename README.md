**Human-Like Battery Usage Simulator**

This repository contains a Python-based simulation of a battery undergoing realistic "human-like" usage patterns over multiple days. The simulation models daily charging cycles (nighttime charging) and various random discharge events (daytime usage), reflecting how a real user might interact with an energy storage device, such as an electric vehicle battery or a home energy system.

**Key Features:**

- **State of Charge (SoC):**  
  Tracks battery SoC over time, adjusting for both charging and discharging events. Charging rate is tapered near full capacity, and discharging power is limited as the battery nears depletion for a more authentic behavior.

- **Capacity Degradation:**  
  Each full cycle (100% to 0% to 100%) reduces the battery’s capacity by a small, configurable degradation rate. Over multiple simulated days, you can observe how the effective capacity gradually declines.

- **Thermal Model:**  
  The simulator includes a simplified but more physically plausible thermal model. It accounts for I²R losses (internal resistance) as a source of heat, a defined thermal mass (representing how much energy is required to change the battery’s temperature), and Newtonian cooling to ambient conditions. This results in more realistic temperature dynamics than simple incremental models.

- **Voltage Modeling:**  
  The pack voltage is approximated based on SoC, linearly interpolating between a defined minimum and maximum voltage per cell.

- **Adjustable Parameters:**  
  Tune parameters to match different battery technologies, capacities, thermal masses, internal resistances, and environmental conditions.

- **Plotting & Visualization:**  
  Automatically generates plots of SoC, capacity, voltage, and temperature over time, making it easy to visualize how the battery responds to the simulated usage profile.

**How It Works:**

1. **Initialization:**  
   Set initial battery conditions, including capacity, charging/discharging rates, efficiencies, thermal characteristics, and ambient conditions.

2. **Daily Simulation:**  
   Each simulated "day" includes a night charging period followed by random usage events throughout the day. Events vary in timing, duration, and intensity, providing a more stochastic and realistic use pattern.

3. **Thermal and Electrical Updates:**  
   After each charging, discharging, or idle step, the simulator updates SoC, temperature, voltage, and capacity. Temperature updates are based on the generated heat (from current flow and internal resistance) and subsequent cooling toward ambient temperature.

4. **Multi-Day Run:**  
   Run the simulation over multiple days to track long-term trends like capacity fade, cycling behavior, and temperature fluctuations.

**Example:**

```python
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
    thermal_mass=5000.0,
    internal_resistance=0.01,
    cooling_coefficient=0.1,
    time_step=0.5
)

# Simulate 10 days of human-like usage
battery.simulate(days=10)

# Plot the results
battery.plot_results()
```

**Applications:**

- **Battery Research & Development:**  
  Quickly test how different battery parameters or user behaviors affect battery life and thermal conditions.

- **Energy Storage Systems Analysis:**  
  Model how daily usage (solar charging, nighttime charging, daily load variations) influences battery performance and longevity.

- **Educational Tool:**  
  Understand the interplay between charging, discharging, temperature, and degradation in a simplified and accessible manner.

---

This simulator is intended as a starting point. It’s deliberately simplified, and real-world batteries are far more complex. Feel free to fork, modify, or extend the model to incorporate more detailed thermal dynamics, aging mechanisms, or more nuanced control strategies.
