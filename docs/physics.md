# ⚛️ Physics Engine

The Heat Treatment Digital Twin simulates precipitation hardening using three coupled Ordinary Differential Equations (ODEs), solved continuously by SciPy's `solve_ivp` (RK45, max step = 120 s). The state vector at every integration step is:

$$\mathbf{y} = [T_{mat},\; r,\; ox]$$

where $T_{mat}$ is the material core temperature (°C), $r$ is the precipitate radius (nm), and $ox$ is the oxidation insulation factor (dimensionless, 0-0.8).

---

## 1. Heat Transfer — Newton's Law of Cooling

The furnace air temperature ($T_{furnace}$) changes **instantaneously** when the agent selects a temperature action, plus Gaussian noise $\mathcal{N}(0, \sigma_T)$. But the material's core temperature follows Newton's Law of Cooling:

$$\frac{dT_{mat}}{dt} = \frac{h_{eff}(t) A_{surface}}{m C_p} (T_{furnace} - T_{mat})$$

| Symbol | Meaning | Source |
|--------|---------|--------|
| $h_{eff}(t)$ | Effective heat transfer coefficient (W/m² K) | `base_h × (1 − ox)` — decays with oxidation |
| $A_{surface}$ | Billet surface area: $2\pi r_b(r_b + h_b)$ (m²) | Computed from `hardware.json` geometry |
| $m$ | Mass: $\rho \pi r_b^2 h_b$ (kg) | `density_g_cm3` × 1000 × volume |
| $C_p$ | Specific heat capacity (J/kg K) | `materials.json` -> `specific_heat_capacity` |

### Why This Matters

This ODE creates **thermal inertia** (lag). A massive casting ($50\text{cm} \times 200\text{cm}$) has orders-of-magnitude more thermal mass than a lab sample ($1\text{cm} \times 5\text{cm}$), so its core temperature takes far longer to equilibrate. The agent must learn **"Predictive Braking"** — cutting the furnace heat long before the material reaches the target temperature — to prevent residual heat from causing catastrophic damage.

### Worked Example: Thermal Mass Comparison

Consider `Ti_6Al_4V` ($\rho = 4.43\text{ g/cm}^3$, $C_p = 526\text{ J/kg K}$) across two hardware setups:

| Property | `lab_scale` (1cm × 5cm) | `massive_casting` (50cm × 200cm) |
|----------|-------------------------|----------------------------------|
| Volume | $\pi 0.01^2 0.05 = 1.57 \times 10^{-5}\text{ m}^3$ | $\pi 0.5^2 2.0 = 1.571\text{ m}^3$ |
| Mass | $0.070\text{ kg}$ | $6{,}959\text{ kg}$ |
| Surface area | $0.00377\text{ m}^2$ | $7.854\text{ m}^2$ |
| $h_{eff} A / (m C_p)$ | $\approx 2.56\text{ s}^{-1}$ | $\approx 1.61 \times 10^{-4}\text{ s}^{-1}$ |
| **Thermal time constant** | **~0.4 s** (near-instant) | **~1.7 hours** (extremely sluggish) |

The massive casting requires ~15,000× longer to equilibrate. This is why the `hard-bake` task (Ti-6Al-4V + massive_casting) demands aggressive predictive braking.

---

## 2. Dynamic Oxidation Kinetics — Arrhenius Insulation

As the material heats, a surface oxide layer builds up, reducing the effective heat transfer coefficient and acting as a thermal insulator. Oxidation follows Arrhenius kinetics:

$$\frac{d(ox)}{dt} = A_{ox} \exp\left(-\frac{E_{ox}}{R (T_{mat} + 273.15)}\right) (0.8 - ox)$$

| Symbol | Meaning | Source |
|--------|---------|--------|
| $A_{ox}$ | Pre-exponential factor for oxidation (1/s) | `materials.json` -> `A_ox` |
| $E_{ox}$ | Activation energy for oxidation (J/mol) | `materials.json` -> `E_ox` |
| $R$ | Universal gas constant = 8.314 J/(mol K) | Constant |
| $(0.8 - ox)$ | Saturation term — caps insulation at 80% | — |

### Saturation Behavior

- The $(0.8 - ox)$ term acts as a self-limiting brake. As the oxide layer thickens, its growth rate slows asymptotically toward zero.
- The effective heat transfer coefficient becomes: $h_{eff} = base\_h (1.0 - ox)$
- At maximum oxidation ($ox = 0.8$), only 20% of the original heat transfer remains. The material becomes increasingly difficult to heat *or* cool.

### Alloy Sensitivity

Different alloys oxidize at vastly different rates:

| Alloy | $A_{ox}$ | $E_{ox}$ (kJ/mol) | Behavior |
|-------|----------|---------------------|----------|
| Inconel 718 | 5 | 180 | Oxidation-resistant superalloy — negligible insulation |
| Al-2024 | 15 | 130 | Moderate oxidation |
| Ti-6Al-4V | 80 | 120 | Fast oxidation at high temperature |
| Mg AZ31B | 100 | 110 | Extremely rapid oxide buildup |

---

## 3. Precipitate Growth — Arrhenius + Phase Thresholds

The base reaction rate $k(T)$ for precipitate growth is driven by the Arrhenius equation:

$$k(T) = A \exp\left(-\frac{E}{R (T_{mat} + 273.15)}\right)$$

| Symbol | Meaning | Source |
|--------|---------|--------|
| $A$ | Pre-exponential factor (reactions/s) | `materials.json` -> `A` |
| $E$ | Activation energy (J/mol) | `materials.json` -> `E` |

### Phase-Dependent Growth Rate

The actual growth rate $dr/dt$ depends on the current thermal regime, defined relative to the alloy's melting temperature ($T_{melt}$):

| Phase | Temperature Range | Growth Rate ($dr/dt$) | Physics |
|-------|-------------------|----------------------|---------|
| **Frozen** | $T < 0.35T_{melt}$ | $0$ | Atomic diffusion is negligible. |
| **Controlled Growth** | $0.35T_{melt} \leq T \leq 0.68T_{melt}$ | $k(T)(1 - r/R_{max})$ | Diffusion-controlled growth. |
| **Ostwald Ripening** | $0.68T_{melt} < T \leq T_{melt}$ | $k(T)(r/R_{max})(1 - r/R_{max})$ | Grain coarsening. Failure mode. |
| **Melting** | $T > T_{melt}$ | $0$ | Crystalline structure dissolves. |

Where $R_{max}$ = `alloy.r_max_clip` — the physical ceiling radius for the alloy.

> **ODE Stability**: All state variables are clamped inside the derivative function ($r \in [0, R_{max}]$, $ox \in [0, 0.8]$) and after ODE integration. This prevents numerical blowup from the positive-feedback ripening term. Additionally, if $r \geq R_{max}$, growth is forced to zero.

### The Natural "Parking Brake"

In the **Controlled Growth** phase, the saturation factor $(1 - r/R_{max})$ creates an emergent deceleration:

- When $r \ll R_{max}$: growth is fast ($dr/dt \approx k(T)$)
- When $r \to R_{max}$: growth slows to near-zero ($dr/dt \to 0$)

This allows the agent to "park" the precipitate radius at the target by holding the material in the Growth phase. In the Ripening phase, the additional $(1 - r/R_{max})$ saturation term prevents numerical blowup while maintaining the physically correct positive-feedback characteristic of Ostwald ripening.

### Phase Thresholds by Alloy

| Alloy | $T_{melt}$ | Frozen ($< 0.35T_m$) | Growth ($0.35-0.68T_m$) | Ripening ($0.68-1.0T_m$) |
|-------|------------|----------------------|-------------------------|--------------------------|
| Al-2024 | 502 °C | < 176 °C | 176-341 °C | 341-502 °C |
| Steel 1095 | 1400 °C | < 490 °C | 490-952 °C | 952-1400 °C |
| Ti-6Al-4V | 1600 °C | < 560 °C | 560-1088 °C | 1088-1600 °C |
| Inconel 718 | 1336 °C | < 468 °C | 468-909 °C | 909-1336 °C |

---

## 4. Reward Model

The reward function shapes the agent's policy toward precision, efficiency, and safety. All server-side rewards are clamped to $[-500, +500]$ to prevent float overflow from corrupting RL gradients.

### Per-Step Reward

At every step with duration $\Delta t$ seconds:

$$R_{step} = -0.1 |r - r_{target}| - 0.01 (r - r_{target})^2 - 0.001 T_{mat} \frac{\Delta t}{3600} - 0.00028 \Delta t$$

Additionally, if $T_{mat}$ exceeds $T_{melt} - 100$°C:

$$R_{step} -= (T_{mat} - T_{warning}) 0.05 \frac{\Delta t}{3600}$$

### Terminal Rewards

| Condition | Reward |
|-----------|--------|
| **Success** ($r_{min} \leq r \leq r_{max}$) | $+100 + 100 \exp\left(-\frac{(r - r_{target})^2}{10}\right)$ |
| **Over-coarsened** ($r > r_{max}$) | $-100$ |
| **Melted** ($T \geq T_{melt}$) | $-200$ |
| **Timed out / Other** | $-50$ |

### Warning Temperature Thresholds

The per-step penalty intensifies when $T_{mat}$ exceeds $T_{melt} - 100$°C. Computed per alloy:

| Alloy | $T_{melt}$ (°C) | $T_{warning}$ (°C) | Penalty kicks in at |
|-------|-----------------|---------------------|---------------------|
| Al-2024 | 502 | 402 | 80% of $T_{melt}$ |
| Mg AZ31B | 630 | 530 | 84% of $T_{melt}$ |
| Inconel 718 | 1336 | 1236 | 93% of $T_{melt}$ |
| Cantor Alloy | 1334 | 1234 | 93% of $T_{melt}$ |
| Steel 1095 | 1400 | 1300 | 93% of $T_{melt}$ |
| Ti-6Al-4V | 1600 | 1500 | 94% of $T_{melt}$ |

### Episode Termination Conditions

An episode ends (`done=True`) when **any** of these conditions are met:

1. **Melting**: $T_{mat} \geq T_{melt}$ — catastrophic material failure (reward: $-200$)
2. **Over-coarsening**: $r > r_{max\_clip}$ — precipitate radius exceeds absolute maximum (reward: $-100$)
3. **Timeout**: $t \geq 180{,}000\text{ s}$ (50 hours) — maximum allowed furnace time exceeded (reward: $-50$)
4. **Agent termination**: `action_num = 5` — the agent voluntarily ends the episode (reward depends on final radius)

---

## 5. ODE Solver Integration

The three coupled ODEs are solved together by `scipy.integrate.solve_ivp`:

```python
solution = solve_ivp(
 fun=self._physics_derivatives, # [dT/dt, dr/dt, d(ox)/dt]
 t_span=(t_current, t_current + duration_sec),
 y0=[T_material, radius, oxidation_factor],
 method='RK45', # Explicit Runge-Kutta order 5(4)
 max_step=120 # Evaluate every 2 simulated minutes
)
```

The solver adaptively refines its internal timestep to maintain accuracy, while `max_step=120` ensures the physics is never integrated over excessively large intervals (preventing missed phase transitions).

### State Clamping

After each integration:

- $r \geq 0$ (numerical noise can produce tiny negatives)
- $ox \leq 0.8$ (enforced saturation cap)
- $T_{furnace}$ is clipped to $[T_{init}, T_{max}]$

---

## 6. Noise & Stochasticity

The furnace temperature includes additive Gaussian noise controlled by the difficulty level:

$$T_{furnace} \leftarrow \text{clip}\left(T_{furnace} + dT_{action} + \mathcal{N}(0, \sigma_T),\; T_{init},\; T_{max}\right)$$

| `AgentGrade` | $\sigma_T$ |
|--------------|-----------|
| EASY | 1 °C |
| MEDIUM | 2 °C |
| HARD | 3 °C |

This noise simulates real-world furnace variability (thermocouple drift, convection turbulence, draft effects).

---

## Summary of Coupled Dynamics

```mermaid
graph TD
 A["Agent Action<br/>(ΔT_furnace, duration)"] -->|Instant + Noise| B["T_furnace"]
 B -->|Newton's Cooling| C["T_material (ODE 1)"]
 C -->|Arrhenius| D["Oxidation ox (ODE 3)"]
 D -->|Reduces h_eff| B
 C -->|Phase-dependent<br/>Arrhenius| E["Precipitate r (ODE 2)"]
 E -->|Proximity reward| F["Reward Signal"]
 C -->|Melt/Overshoot<br/>penalties| F
```

The key insight is the **feedback loop**: oxidation reduces heat transfer, which affects temperature evolution, which in turn affects both oxidation rate and precipitate growth. This creates complex, non-linear dynamics that require predictive multi-step planning rather than greedy single-step optimization.
