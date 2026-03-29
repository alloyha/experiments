"""
Ball kicking dynamics simulation in the x-y plane.

Forces acting on the ball:
  - Gravity:       (0, -m*g)
  - Air drag:      F_drag = -½ ρ C_d A |v|² v̂  =  -½ ρ C_d A |v| (vx, vy)
  - Magnus effect: F_mag  =  ½ ρ C_L(S) A |v|² (ω̂ × v̂)

    where the spin parameter S = R·|ω| / |v| controls the lift coefficient
    C_L(S) ≈ C_L0 · S  (linear regime, valid for S < ~0.5).

    In 2-D with ω along z:  ω̂ × v̂ = (−vy, vx) / |v|
    so  F_mag = ½ ρ C_L0 (Rω/|v|) A |v|² (−vy, vx)/|v|
             = ½ ρ C_L0 A R ω (−vy, vx)  / |v|
             ≡ 0  when  |v| → 0  (protected).

Spin-down torque (aerodynamic rotational drag on a spinning sphere):
  I ω̇ = −½ ρ C_R π R³ |ω| ω

  with I = ⅔ m R²  (thin spherical shell — good model for a soccer ball).

ρ(x,y) and C_d(x,y) are parametrized as callables for future development,
but set as constants for now.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable
from itertools import product


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class BallParams:
    mass: float = 0.45           # kg  (FIFA size-5 ball)
    radius: float = 0.11         # m
    g: float = 9.81              # m/s²

    # Drag coefficient (dimensionless)
    C_d: float = 0.47            # typical sphere / soccer ball

    # Magnus / lift
    C_L0: float = 1.2            # lift coefficient slope:  C_L(S) = C_L0 * S

    # Rotational (spin-down) drag coefficient (dimensionless)
    C_R: float = 0.03            # aero torque on spinning sphere

    # Spatial parametrizations — callable(x, y) -> float
    # Default: constants everywhere
    rho: Callable[[float, float], float] = field(
        default_factory=lambda: lambda x, y: 1.225   # kg/m³ (sea-level air)
    )
    drag_coeff: Callable[[float, float], float] = field(
        default_factory=lambda: lambda x, y: 0.47    # C_d(x,y) — same as C_d above
    )

    @property
    def A(self) -> float:
        """Cross-sectional area π R²."""
        return np.pi * self.radius**2

    @property
    def I(self) -> float:
        """Moment of inertia — thin spherical shell: ⅔ m R²."""
        return (2 / 3) * self.mass * self.radius**2


# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------
def equations_of_motion(t, state, params: BallParams):
    """
    state = [x, y, vx, vy, omega]
    Returns d(state)/dt.
    """
    x, y, vx, vy, omega = state
    speed = np.hypot(vx, vy)

    rho = params.rho(x, y)
    C_d = params.drag_coeff(x, y)
    m = params.mass
    R = params.radius
    A = params.A

    # --- Translational drag:  F_drag = -½ ρ C_d A |v| (vx, vy) -----------
    half_rho_A = 0.5 * rho * A
    drag_x = -half_rho_A * C_d * speed * vx
    drag_y = -half_rho_A * C_d * speed * vy

    # --- Magnus force -----------------------------------------------------
    # F_mag = ½ ρ C_L(S) A |v|²  (ω̂ × v̂)
    # With C_L(S) = C_L0 · S = C_L0 · R|ω|/|v| and (ω̂ × v̂) = (-vy, vx)/|v|
    # => F_mag = ½ ρ C_L0 A R ω (-vy, vx) / |v|   (when |v| > 0)
    if speed > 1e-12:
        magnus_fac = half_rho_A * params.C_L0 * R * omega / speed
        magnus_x = magnus_fac * (-vy)
        magnus_y = magnus_fac * vx
    else:
        magnus_x = 0.0
        magnus_y = 0.0

    ax = (drag_x + magnus_x) / m
    ay = -params.g + (drag_y + magnus_y) / m

    # --- Spin-down torque:  I ω̇ = -½ ρ C_R π R³ |ω| ω --------------------
    torque = -0.5 * rho * params.C_R * np.pi * R**3 * abs(omega) * omega
    omega_dot = torque / params.I

    return [vx, vy, ax, ay, omega_dot]


# ---------------------------------------------------------------------------
# Ground-hit event (y = 0 after launch)
# ---------------------------------------------------------------------------
def hit_ground(t, state, params):
    return state[1]  # y coordinate

hit_ground.terminal = True
hit_ground.direction = -1  # falling through y = 0


# ---------------------------------------------------------------------------
# Run a single trajectory
# ---------------------------------------------------------------------------
def simulate(
    v0: float,
    theta: float,
    omega: float,
    params: BallParams | None = None,
    t_max: float = 15.0,
    dt: float = 0.02,
):
    """
    Parameters
    ----------
    v0    : initial linear speed (m/s)
    theta : launch angle (radians)
    omega : angular velocity (rad/s), positive = backspin
    params: ball parameters
    t_max : maximum integration time (s)
    dt    : output time step (s)

    Returns
    -------
    sol   : OdeSolution from solve_ivp
    """
    if params is None:
        params = BallParams()

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    state0 = [0.0, 0.0, vx0, vy0, omega]

    t_eval = np.arange(0, t_max, dt)

    sol = solve_ivp(
        equations_of_motion,
        [0, t_max],
        state0,
        args=(params,),
        method="RK45",
        t_eval=t_eval,
        events=hit_ground,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol


# ---------------------------------------------------------------------------
# Parameter sweep & plotting
# ---------------------------------------------------------------------------
def run_sweep():
    theta = np.radians(30)  # fixed launch angle
    params = BallParams()

    linear_velocities = np.arange(10, 55, 10)   # 10, 20, 30, 40, 50 m/s
    angular_velocities = np.arange(10, 55, 10)   # 10 .. 50 rad/s

    # --- Figure 1: vary v0, fixed omega ---------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    omega_showcase = [10, 30, 50]
    for ax, omega_fix in zip(axes1, omega_showcase):
        for v0 in linear_velocities:
            sol = simulate(v0, theta, omega_fix, params)
            x, y = sol.y[0], sol.y[1]
            mask = y >= 0
            ax.plot(x[mask], y[mask], label=f"v₀={v0} m/s")
        ax.set_title(f"ω = {omega_fix} rad/s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    fig1.suptitle(
        f"Ball trajectories — varying linear velocity (θ = {np.degrees(theta):.0f}°)",
        fontsize=14,
    )
    fig1.tight_layout()
    fig1.savefig("trajectories_vary_v0.png", dpi=150)

    # --- Figure 2: vary omega, fixed v0 ---------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    v0_showcase = [10, 30, 50]
    for ax, v0_fix in zip(axes2, v0_showcase):
        for omega in angular_velocities:
            sol = simulate(v0_fix, theta, omega, params)
            x, y = sol.y[0], sol.y[1]
            mask = y >= 0
            ax.plot(x[mask], y[mask], label=f"ω={omega} rad/s")
        ax.set_title(f"v₀ = {v0_fix} m/s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    fig2.suptitle(
        f"Ball trajectories — varying angular velocity (θ = {np.degrees(theta):.0f}°)",
        fontsize=14,
    )
    fig2.tight_layout()
    fig2.savefig("trajectories_vary_omega.png", dpi=150)

    # --- Figure 3: full grid heatmap of range ----------------------------
    ranges = np.zeros((len(linear_velocities), len(angular_velocities)))
    max_heights = np.zeros_like(ranges)

    for i, v0 in enumerate(linear_velocities):
        for j, omega in enumerate(angular_velocities):
            sol = simulate(v0, theta, omega, params)
            x, y = sol.y[0], sol.y[1]
            mask = y >= 0
            ranges[i, j] = x[mask][-1] if mask.any() else 0.0
            max_heights[i, j] = y[mask].max() if mask.any() else 0.0

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax3a.imshow(
        ranges,
        origin="lower",
        aspect="auto",
        extent=[
            angular_velocities[0] - 5,
            angular_velocities[-1] + 5,
            linear_velocities[0] - 5,
            linear_velocities[-1] + 5,
        ],
    )
    ax3a.set_xlabel("ω (rad/s)")
    ax3a.set_ylabel("v₀ (m/s)")
    ax3a.set_title("Horizontal range (m)")
    fig3.colorbar(im1, ax=ax3a)

    im2 = ax3b.imshow(
        max_heights,
        origin="lower",
        aspect="auto",
        extent=[
            angular_velocities[0] - 5,
            angular_velocities[-1] + 5,
            linear_velocities[0] - 5,
            linear_velocities[-1] + 5,
        ],
    )
    ax3b.set_xlabel("ω (rad/s)")
    ax3b.set_ylabel("v₀ (m/s)")
    ax3b.set_title("Maximum height (m)")
    fig3.colorbar(im2, ax=ax3b)

    fig3.suptitle(
        f"Sweep over v₀ and ω (θ = {np.degrees(theta):.0f}°)", fontsize=14
    )
    fig3.tight_layout()
    fig3.savefig("sweep_heatmaps.png", dpi=150)

    # --- Figure 4: time-series for a sample case -------------------------
    fig4, axes4 = plt.subplots(3, 2, figsize=(12, 11))
    sample_v0, sample_omega = 30.0, 30.0
    sol = simulate(sample_v0, theta, sample_omega, params)
    t = sol.t
    x, y, vx, vy, omega_t = sol.y
    mask = y >= 0

    axes4[0, 0].plot(t[mask], x[mask])
    axes4[0, 0].set_ylabel("x (m)")
    axes4[0, 0].set_title("Horizontal position")

    axes4[0, 1].plot(t[mask], y[mask])
    axes4[0, 1].set_ylabel("y (m)")
    axes4[0, 1].set_title("Vertical position")

    axes4[1, 0].plot(t[mask], vx[mask], label="vx")
    axes4[1, 0].plot(t[mask], vy[mask], label="vy")
    axes4[1, 0].set_ylabel("velocity (m/s)")
    axes4[1, 0].set_xlabel("t (s)")
    axes4[1, 0].set_title("Velocity components")
    axes4[1, 0].legend()

    speed = np.hypot(vx, vy)
    axes4[1, 1].plot(t[mask], speed[mask])
    axes4[1, 1].set_ylabel("|v| (m/s)")
    axes4[1, 1].set_xlabel("t (s)")
    axes4[1, 1].set_title("Speed magnitude")

    axes4[2, 0].plot(t[mask], omega_t[mask], color="tab:red")
    axes4[2, 0].set_ylabel("ω (rad/s)")
    axes4[2, 0].set_xlabel("t (s)")
    axes4[2, 0].set_title("Angular velocity (spin decay)")

    # Spin parameter S = R|ω|/|v|
    S = params.radius * np.abs(omega_t) / np.maximum(speed, 1e-12)
    axes4[2, 1].plot(t[mask], S[mask], color="tab:purple")
    axes4[2, 1].set_ylabel("S = Rω/|v|")
    axes4[2, 1].set_xlabel("t (s)")
    axes4[2, 1].set_title("Spin parameter")

    for ax in axes4.flat:
        ax.grid(True, alpha=0.3)
    fig4.suptitle(
        f"Time series — v₀={sample_v0} m/s, ω₀={sample_omega} rad/s, "
        f"θ={np.degrees(theta):.0f}°",
        fontsize=13,
    )
    fig4.tight_layout()
    fig4.savefig("time_series_sample.png", dpi=150)

    plt.close("all")
    print("Figures saved.")


if __name__ == "__main__":
    run_sweep()
