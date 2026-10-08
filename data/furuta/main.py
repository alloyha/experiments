"""
Furuta Rotary Inverted Pendulum
===============================

Pipeline:

1. Planta física completa:
       x = [theta, alpha, theta_dot, alpha_dot, current]

2. PWM explícito:
       F_PWM = 20 kHz
       sign-magnitude ideal
       ON  = +/- Vdc
       OFF = 0 V

3. Dinâmica elétrica:
       resolvida ANALITICAMENTE em cada trecho ON/OFF.

4. Mecânica:
       integrada apenas uma vez por período de controle Ts,
       usando torque médio produzido pelo PWM.

5. Modelo reduzido para estimação:
       x_red = [theta, alpha_error, theta_dot, alpha_dot]

       com corrente quasi-estática.

6. Zero dynamics:
       obtida do modelo contínuo médio completo via Rosenbrock.

7. Controle:
       LQI tangencial-transversal discreto com integrador de theta.

8. Estimação:
       Kalman discreto usando somente theta.

Objetivo:
       alpha -> pi
       alpha_dot -> 0
       theta -> 0 (absolute)
       theta_dot -> 0

O integrador de theta elimina erro estacionario de posicao
do braco, com anti-windup durante saturacao.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import sympy as sp

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.linalg import (
    eig,
    null_space,
    block_diag,
    orth,
    solve_discrete_are,
)

from scipy.signal import (
    cont2discrete,
    place_poles,
)
from scipy.optimize import differential_evolution

from scipy.linalg import (
    eig,
    null_space,
    block_diag,
    orth,
    solve_discrete_are,
    expm,
)


np.set_printoptions(
    precision=10,
    suppress=True,
)


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

T_SAMPLE = 0.002
F_SAMPLE = 1.0 / T_SAMPLE

SIMULATION_TIME = 8.0

MAX_DUTY = 1.0

SWING_UP_SWITCH_ANGLE = np.deg2rad(25.0)
# Tightened from 360 deg/s: basin-of-attraction testing (see debug notes)
# shows the corrected LQI reliably recovers for handoff angular rates up
# to ~180-200 deg/s across the whole +/-25 deg catch window, but can fail
# right at 300-360 deg/s. 180 deg/s keeps every tested combination well
# inside the verified capture basin.
SWING_UP_SWITCH_ALPHA_RATE = np.deg2rad(180.0)

# If the LQI is engaged but the pendulum strays past this angle from the
# top, it has lost the catch: the linear controller has no validity that
# far from equilibrium, so we disengage and let the swing-up pump
# re-establish energy rather than let the LQI fight (and fail) in a
# regime it was never designed for.
SWING_UP_REENGAGE_ANGLE = np.deg2rad(45.0)

# Hysteresis band on the energy error (in Joules) around zero, used only to
# stop duty chatter once the pendulum is essentially at the target energy.
SWING_UP_ENERGY_DEADBAND = 0.005

# ============================================================
# ONE-SWING SYNTHESIS
# ============================================================
# The planned swing is restricted to one alternating, finite sequence.
# The optimizer chooses the switching durations while enforcing:
#
#       theta(T) ~= 0
#       theta_dot(T) small
#       |theta(t)| <= 2*pi  (360 deg)
#       terminal pendulum state inside the LQI capture window.
#
# If no feasible plan is found, the simulation aborts instead of using
# an unconstrained energy-pump fallback that could violate the arm limit.

ONE_SWING_ENABLE = True

# False: use the already synthesized/validated durations below.
# True : rerun differential evolution when the script starts.
ONE_SWING_REDESIGN = False

ONE_SWING_SIGNS = np.array([
    -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0,
])

# Nominal solution synthesized for the physical parameters in this file.
# Total duration ~= 3.15050 s.
ONE_SWING_NOMINAL_DURATIONS = np.array([
    0.5108844491,
    0.1635736562,
    0.0971325810,
    0.7527305436,
    0.4824041240,
    0.4989880832,
    0.3860485780,
    0.1528992686,
    0.1058425247,
])

ONE_SWING_DURATION_BOUNDS = (0.015, 0.90)
ONE_SWING_DESIGN_DT = 0.004
ONE_SWING_MAXITER = 120
ONE_SWING_POPSIZE = 12
ONE_SWING_SEEDS = (7, 17, 29, 41)

# Terminal weights/scales for the one-swing shooting problem.
ONE_SWING_TARGET_ANGLE = np.deg2rad(8.0)
ONE_SWING_TARGET_ALPHA_RATE = np.deg2rad(70.0)
ONE_SWING_TARGET_THETA_RATE = np.deg2rad(45.0)

# Hard geometric requirements for the planned swing.
ONE_SWING_HANDOFF_THETA_LIMIT = np.deg2rad(280.0)
ONE_SWING_THETA_LIMIT = np.deg2rad(360.0)

# Extra handoff safety constraint.  Alpha and alpha_dot are already
# constrained by SWING_UP_SWITCH_* above; this prevents handing a very
# fast arm to the local LQI even if the pendulum happens to pass the top.
SWING_UP_SWITCH_THETA_RATE = np.deg2rad(140.0)

# If the nominal one-swing misses the verified capture window because the
# model/parameters were changed, fall back to the original energy pump.
ONE_SWING_FALLBACK_TO_ENERGY = False


# ============================================================
# 2. PWM
# ============================================================

F_PWM = 20_000.0
T_PWM = 1.0 / F_PWM


PWM_CYCLES_PER_CONTROL = int(
    round(
        F_PWM / F_SAMPLE
    )
)


if not np.isclose(
    PWM_CYCLES_PER_CONTROL * T_PWM,
    T_SAMPLE,
):
    raise RuntimeError(
        "F_PWM must be an integer multiple of F_SAMPLE."
    )


# ============================================================
# 3. MOTOR
# ============================================================

V_DC = 12.0

R_MOTOR = 4.0
L_MOTOR = 8.0e-3

K_T = 0.040
K_E = 0.040

# NOTE (debug fix): with GEAR_RATIO = 1.0 the motor can only sustain about
# Kt*Vdc/R = 0.04*12/4 = 0.12 N.m at the arm. Diagnostics (see debug notes)
# show that even a torque/energy-optimal bang-bang swing-up policy driven
# with unbounded time (and even with friction removed) plateaus around an
# alpha amplitude of ~110-120 deg with that torque budget: the motor is
# simply too weak to invert this pendulum, direct-drive, by itself -- this
# is a genuine actuator-authority limit, not a controller tuning issue.
# A modest 3:1 gearbox (a completely standard fix on real Furuta rigs)
# triples torque at the arm and is sufficient for swing-up to succeed
# reliably in well under 2 seconds with the corrected bang-bang law below.
GEAR_RATIO = 3.0
GEAR_EFFICIENCY = 0.85


TAU_ELECTRICAL = (
    L_MOTOR / R_MOTOR
)


# ============================================================
# 4. CONTROLADOR UNIFICADO TANGENCIAL-TRANSVERSAL (LQI)
#
# Coordenadas do modelo linear:
#
#       xi = [eta ; z_perp]
#
# e adicionamos um estado integral da posicao do braco:
#
#       zeta_theta[k+1] = zeta_theta[k] + Ts * theta_hat[k]
#
# O controlador final e um unico ganho LQI:
#
#       duty = -K_XI @ xi_hat - K_I * zeta_theta
#
# A parte transversal recebe peso muito maior que a parte
# tangencial. O integrador elimina erro estacionario em theta.
# Como ha saturacao |duty| <= 1, usamos anti-windup por
# integracao condicional no loop de simulacao.
# ============================================================

Q_ETA_WEIGHT = np.diag([
    1.0,      # eta_1 ~ theta
    1.0,      # eta_2 ~ theta_dot
])

Q_ZPERP_WEIGHT = np.diag([
    1.0e4,    # alta prioridade transversal
    1.0e2,
    1.0e2,
])

# Peso do estado integral de theta.
# Aumentar -> elimina offset mais agressivamente, mas pode
# elevar duty e saturacao.
Q_INTEGRAL_WEIGHT = 25.0

R_XI = np.array([
    [1.0]
])

# Limite numerico adicional para o estado integral.
# O anti-windup principal e condicional, mas este clamp evita
# crescimento ilimitado em casos patologicos.
THETA_INTEGRAL_LIMIT = np.deg2rad(120.0) * 10.0


# ============================================================
# 5. KALMAN
# ============================================================

Q_EST = np.diag([
    1e-6,
    1e-3,
    1e-2,
    1e-2,
])


R_EST = np.array([
    [1e-5]
])


# ============================================================
# 6. VARIÁVEIS SIMBÓLICAS
# ============================================================

theta, alpha = sp.symbols(
    "theta alpha",
    real=True,
)

theta_dot, alpha_dot = sp.symbols(
    "theta_dot alpha_dot",
    real=True,
)

theta_ddot, alpha_ddot = sp.symbols(
    "theta_ddot alpha_ddot",
    real=True,
)

tau_arm_symbol = sp.symbols(
    "tau_arm",
    real=True,
)


m, Lr, Lp = sp.symbols(
    "m Lr Lp",
    positive=True,
)

Jr, Jp = sp.symbols(
    "Jr Jp",
    positive=True,
)

br, bp = sp.symbols(
    "br bp",
    nonnegative=True,
)

g = sp.symbols(
    "g",
    positive=True,
)


q = sp.Matrix([
    theta,
    alpha,
])

qd = sp.Matrix([
    theta_dot,
    alpha_dot,
])

qdd = sp.Matrix([
    theta_ddot,
    alpha_ddot,
])


# ============================================================
# 7. GEOMETRIA
# ============================================================

x_p = (
    Lr * sp.cos(theta)
    - Lp * sp.sin(alpha) * sp.sin(theta)
)

y_p = (
    Lr * sp.sin(theta)
    + Lp * sp.sin(alpha) * sp.cos(theta)
)

z_p = (
    -Lp * sp.cos(alpha)
)


position = sp.Matrix([
    x_p,
    y_p,
    z_p,
])


velocity = (
    position.jacobian(q)
    @ qd
)


# ============================================================
# 8. ENERGIA
# ============================================================

T_trans = (
    sp.Rational(1, 2)
    * m
    * velocity.dot(velocity)
)

T_arm = (
    sp.Rational(1, 2)
    * Jr
    * theta_dot**2
)

T_pend = (
    sp.Rational(1, 2)
    * Jp
    * alpha_dot**2
)


T = (
    T_trans
    + T_arm
    + T_pend
)


V = (
    m
    * g
    * z_p
)


Lagrangian = (
    T - V
)


# ============================================================
# 9. EULER-LAGRANGE
# ============================================================

def total_time_derivative(expr):

    return (
        sp.diff(expr, theta)
        * theta_dot

        + sp.diff(expr, alpha)
        * alpha_dot

        + sp.diff(expr, theta_dot)
        * theta_ddot

        + sp.diff(expr, alpha_dot)
        * alpha_ddot
    )


dL_dqd = sp.Matrix([
    sp.diff(
        Lagrangian,
        theta_dot,
    ),

    sp.diff(
        Lagrangian,
        alpha_dot,
    ),
])


dL_dq = sp.Matrix([
    sp.diff(
        Lagrangian,
        theta,
    ),

    sp.diff(
        Lagrangian,
        alpha,
    ),
])


dt_dL_dqd = sp.Matrix([
    total_time_derivative(
        dL_dqd[0]
    ),

    total_time_derivative(
        dL_dqd[1]
    ),
])


generalized_forces = sp.Matrix([
    tau_arm_symbol
    - br * theta_dot,

    -bp * alpha_dot,
])


EL = (
    dt_dL_dqd
    - dL_dq
    - generalized_forces
)


M_symbolic = (
    EL.jacobian(
        qdd
    )
)


h_symbolic = (
    EL
    - M_symbolic
    @ qdd
)


# ============================================================
# 10. PARÂMETROS MECÂNICOS
# ============================================================

PENDULUM_LENGTH = 0.340
PENDULUM_COM = PENDULUM_LENGTH / 2.0
PENDULUM_MASS = 0.125

PENDULUM_INERTIA_CM = (
    PENDULUM_MASS
    * PENDULUM_LENGTH**2
    / 12.0
)

ARM_LENGTH = 0.215
ARM_MASS = 0.300
HUB_INERTIA = 0.00038

ARM_INERTIA_ROD = (
    ARM_MASS
    * ARM_LENGTH**2
    / 3.0
)

ARM_INERTIA_TOTAL = (
    ARM_INERTIA_ROD
    + HUB_INERTIA
)

params = {
    m: PENDULUM_MASS,

    Lr: ARM_LENGTH,
    Lp: PENDULUM_COM,

    Jr: ARM_INERTIA_TOTAL,
    Jp: PENDULUM_INERTIA_CM,

    br: 0.0020,
    bp: 0.0010,

    g: 9.81,
}


PENDULUM_INERTIA = (
    params[Jp]
    + params[m] * params[Lp] ** 2
)

PENDULUM_ENERGY_TARGET = (
    params[m]
    * params[g]
    * params[Lp]
)


parameter_values = (
    params[m],

    params[Lr],
    params[Lp],

    params[Jr],
    params[Jp],

    params[br],
    params[bp],

    params[g],
)


mechanical_arguments = (

    theta,
    alpha,

    theta_dot,
    alpha_dot,

    tau_arm_symbol,

    m,
    Lr,
    Lp,

    Jr,
    Jp,

    br,
    bp,

    g,
)


M_numeric = sp.lambdify(
    mechanical_arguments,
    M_symbolic,
    "numpy",
)


h_numeric = sp.lambdify(
    mechanical_arguments,
    h_symbolic,
    "numpy",
)


# ============================================================
# 11. UTILIDADES
# ============================================================

def wrap_to_pi(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def saturate_duty(duty):

    return float(
        np.clip(
            duty,
            -MAX_DUTY,
            MAX_DUTY,
        )
    )


def motor_to_arm_torque(current):

    return (
        GEAR_EFFICIENCY
        * GEAR_RATIO
        * K_T
        * current
    )


# ============================================================
# 12. DINÂMICA MECÂNICA
# ============================================================

def mechanical_rhs_torque(
    x_mech,
    torque,
):

    theta_value = x_mech[0]
    alpha_value = x_mech[1]

    theta_dot_value = x_mech[2]
    alpha_dot_value = x_mech[3]


    arguments = (

        theta_value,
        alpha_value,

        theta_dot_value,
        alpha_dot_value,

        torque,

        *parameter_values,
    )


    M_value = np.asarray(
        M_numeric(
            *arguments
        ),
        dtype=float,
    )


    h_value = np.asarray(
        h_numeric(
            *arguments
        ),
        dtype=float,
    ).reshape(2)


    acceleration = np.linalg.solve(
        M_value,
        -h_value,
    )


    return np.array([

        theta_dot_value,

        alpha_dot_value,

        acceleration[0],

        acceleration[1],
    ])


# ============================================================
# 13. RK4 MECÂNICO
# ============================================================

def mechanical_rk4_step(
    x_mech,
    torque,
    dt,
):

    k1 = mechanical_rhs_torque(
        x_mech,
        torque,
    )


    k2 = mechanical_rhs_torque(
        x_mech
        + 0.5 * dt * k1,
        torque,
    )


    k3 = mechanical_rhs_torque(
        x_mech
        + 0.5 * dt * k2,
        torque,
    )


    k4 = mechanical_rhs_torque(
        x_mech
        + dt * k3,
        torque,
    )


    return (
        x_mech

        + dt
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
        / 6.0
    )


# ============================================================
# 14. DINÂMICA ELÉTRICA EXATA
#
# L di/dt =
#       voltage
#       - R i
#       - Ke N theta_dot
#
# voltage e theta_dot são assumidos constantes durante dt.
# ============================================================

def exact_current_segment(
    current,
    theta_dot_value,
    voltage,
    dt,
):

    if dt <= 0.0:

        return (
            current,
            0.0,
        )


    tau_e = (
        L_MOTOR
        / R_MOTOR
    )


    back_emf = (
        K_E
        * GEAR_RATIO
        * theta_dot_value
    )


    i_inf = (
        voltage
        - back_emf
    ) / R_MOTOR


    decay = np.exp(
        -dt / tau_e
    )


    current_next = (
        i_inf

        + (
            current
            - i_inf
        )
        * decay
    )


    # Integral exata da corrente no trecho
    integral_current = (

        i_inf
        * dt

        + (
            current
            - i_inf
        )
        * tau_e
        * (
            1.0
            - decay
        )
    )


    return (
        current_next,
        integral_current,
    )


# ============================================================
# 15. UM CICLO PWM EXATO
# ============================================================

def exact_pwm_cycle(
    current,
    theta_dot_value,
    duty,
):

    duty = saturate_duty(
        duty
    )


    magnitude = abs(
        duty
    )


    ton = (
        magnitude
        * T_PWM
    )


    toff = (
        T_PWM
        - ton
    )


    integral_current = 0.0


    # --------------------------------------------------------
    # ON
    # --------------------------------------------------------

    if ton > 0.0:

        voltage_on = (
            np.sign(duty)
            * V_DC
        )


        (
            current,
            integral_on,

        ) = exact_current_segment(
            current,
            theta_dot_value,
            voltage_on,
            ton,
        )


        integral_current += (
            integral_on
        )


    # --------------------------------------------------------
    # OFF
    # --------------------------------------------------------

    if toff > 0.0:

        (
            current,
            integral_off,

        ) = exact_current_segment(
            current,
            theta_dot_value,
            0.0,
            toff,
        )


        integral_current += (
            integral_off
        )


    return (
        current,
        integral_current,
    )


# ============================================================
# 16. MULTIRATE PLANT STEP
#
# Um período de controle:
#
# 1. duty permanece constante por Ts
# 2. corrente percorre 40 ciclos PWM
# 3. corrente é integrada exatamente
# 4. calcula-se corrente média
# 5. calcula-se torque médio
# 6. mecânica avança uma vez por Ts
# ============================================================

def multirate_plant_step(
    x,
    duty,
    store_pwm=False,
):

    x = np.asarray(
        x,
        dtype=float,
    )


    x_mech = (
        x[:4].copy()
    )


    current = float(
        x[4]
    )


    # theta_dot é considerado aproximadamente constante
    # durante este Ts apenas para a dinâmica elétrica rápida.
    theta_dot_frozen = float(
        x_mech[2]
    )


    total_current_integral = 0.0


    if store_pwm:

        pwm_time = [
            0.0
        ]

        pwm_current = [
            current
        ]

        pwm_voltage = [
            np.nan
        ]


    elapsed = 0.0


    for _ in range(
        PWM_CYCLES_PER_CONTROL
    ):

        duty_clipped = (
            saturate_duty(
                duty
            )
        )


        magnitude = abs(
            duty_clipped
        )


        ton = (
            magnitude
            * T_PWM
        )


        toff = (
            T_PWM
            - ton
        )


        # ----------------------------------------------------
        # ON
        # ----------------------------------------------------

        if ton > 0.0:

            voltage_on = (
                np.sign(
                    duty_clipped
                )
                * V_DC
            )


            (
                current,
                integral_on,

            ) = exact_current_segment(
                current,
                theta_dot_frozen,
                voltage_on,
                ton,
            )


            total_current_integral += (
                integral_on
            )


            elapsed += ton


            if store_pwm:

                pwm_time.append(
                    elapsed
                )

                pwm_current.append(
                    current
                )

                pwm_voltage.append(
                    voltage_on
                )


        # ----------------------------------------------------
        # OFF
        # ----------------------------------------------------

        if toff > 0.0:

            (
                current,
                integral_off,

            ) = exact_current_segment(
                current,
                theta_dot_frozen,
                0.0,
                toff,
            )


            total_current_integral += (
                integral_off
            )


            elapsed += toff


            if store_pwm:

                pwm_time.append(
                    elapsed
                )

                pwm_current.append(
                    current
                )

                pwm_voltage.append(
                    0.0
                )


    # --------------------------------------------------------
    # Média exata da corrente durante Ts
    # --------------------------------------------------------

    current_average = (
        total_current_integral
        / T_SAMPLE
    )


    # --------------------------------------------------------
    # Torque médio
    # --------------------------------------------------------

    torque_average = (
        motor_to_arm_torque(
            current_average
        )
    )


    # --------------------------------------------------------
    # Integração mecânica
    # --------------------------------------------------------

    x_mech_next = (
        mechanical_rk4_step(
            x_mech,
            torque_average,
            T_SAMPLE,
        )
    )


    x_next = np.array([

        x_mech_next[0],

        x_mech_next[1],

        x_mech_next[2],

        x_mech_next[3],

        current,
    ])


    if store_pwm:

        return {

            "x_next":
                x_next,

            "current_average":
                current_average,

            "current_final":
                current,

            "torque_average":
                torque_average,

            "pwm_time":
                np.asarray(
                    pwm_time
                ),

            "pwm_current":
                np.asarray(
                    pwm_current
                ),

            "pwm_voltage":
                np.asarray(
                    pwm_voltage
                ),
        }


    return x_next


# ============================================================
# 17. MODELO COMPLETO MÉDIO
#
# Usado para:
#
# - linearização
# - Rosenbrock
# - síntese
#
# A simulação principal usa multirate_plant_step().
# ============================================================

N_FULL = 5


def full_average_rhs(
    x,
    duty,
):

    theta_value = x[0]
    alpha_value = x[1]

    theta_dot_value = x[2]
    alpha_dot_value = x[3]

    current = x[4]


    torque = (
        motor_to_arm_torque(
            current
        )
    )


    x_mech = np.array([
        theta_value,
        alpha_value,
        theta_dot_value,
        alpha_dot_value,
    ])


    mechanical_dot = (
        mechanical_rhs_torque(
            x_mech,
            torque,
        )
    )


    voltage_average = (
        V_DC
        * duty
    )


    current_dot = (

        voltage_average

        - R_MOTOR
        * current

        - K_E
        * GEAR_RATIO
        * theta_dot_value

    ) / L_MOTOR


    return np.array([

        mechanical_dot[0],

        mechanical_dot[1],

        mechanical_dot[2],

        mechanical_dot[3],

        current_dot,
    ])


# ============================================================
# 18. MODELO REDUZIDO
# ============================================================

N_RED = 4


def quasi_steady_current(
    theta_dot_value,
    duty,
):

    return (

        V_DC
        * duty

        - K_E
        * GEAR_RATIO
        * theta_dot_value

    ) / R_MOTOR


def reduced_rhs(
    x,
    duty,
):

    current = (
        quasi_steady_current(
            x[2],
            duty,
        )
    )


    torque = (
        motor_to_arm_torque(
            current
        )
    )


    return (
        mechanical_rhs_torque(
            x,
            torque,
        )
    )


# ============================================================
# 19. EQUILÍBRIOS
# ============================================================

x_eq_full = np.array([
    0.0,
    np.pi,
    0.0,
    0.0,
    0.0,
])


x_eq_red = np.array([
    0.0,
    np.pi,
    0.0,
    0.0,
])


# ============================================================
# 20. LINEARIZAÇÃO
# ============================================================

def numerical_linearization(
    f,
    x0,
    u0=0.0,
    eps=1e-6,
):

    n = len(
        x0
    )


    A = np.zeros(
        (
            n,
            n,
        )
    )


    B = np.zeros(
        (
            n,
            1,
        )
    )


    for j in range(n):

        dx = np.zeros(
            n
        )


        dx[j] = (
            eps
        )


        A[:, j] = (

            f(
                x0 + dx,
                u0,
            )

            - f(
                x0 - dx,
                u0,
            )

        ) / (
            2.0 * eps
        )


    B[:, 0] = (

        f(
            x0,
            u0 + eps,
        )

        - f(
            x0,
            u0 - eps,
        )

    ) / (
        2.0 * eps
    )


    return (
        A,
        B,
    )


A_full, B_full = (
    numerical_linearization(
        full_average_rhs,
        x_eq_full,
    )
)


A_red, B_red = (
    numerical_linearization(
        reduced_rhs,
        x_eq_red,
    )
)


# ============================================================
# 21. ROSENBROCK
# ============================================================

C_zero = np.array([
    [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
])


D_zero = np.array([
    [0.0]
])


def cluster_values(
    values,
    tolerance=1e-7,
):

    clusters = []


    for value in values:

        assigned = False


        for cluster in clusters:

            representative = (
                np.mean(
                    cluster
                )
            )


            scale = max(
                1.0,
                abs(value),
                abs(representative),
            )


            if (
                abs(
                    value
                    - representative
                )

                <=

                tolerance
                * scale
            ):

                cluster.append(
                    value
                )

                assigned = True

                break


        if not assigned:

            clusters.append([
                value
            ])


    return clusters


def rosenbrock_basis(
    A,
    B,
    C,
    D,
):

    n = (
        A.shape[0]
    )


    E = np.block([

        [
            np.eye(n),

            np.zeros(
                (
                    n,
                    1,
                )
            ),
        ],

        [
            np.zeros(
                (
                    1,
                    n,
                )
            ),

            np.zeros(
                (
                    1,
                    1,
                )
            ),
        ],
    ])


    F = np.block([

        [
            A,
            B,
        ],

        [
            -C,
            -D,
        ],
    ])


    values, _ = eig(
        F,
        E,
    )


    finite = values[
        np.isfinite(
            values
        )
    ]


    clusters = (
        cluster_values(
            finite
        )
    )


    V_blocks = []

    J_blocks = []


    for cluster in clusters:

        zero = (
            np.mean(
                cluster
            )
        )


        multiplicity = (
            len(
                cluster
            )
        )


        K = (
            F
            - zero * E
        )


        kernel = (
            null_space(
                K
            )
        )


        chains = []


        for j in range(
            kernel.shape[1]
        ):

            v = (
                kernel[:, j]
            )


            state_part = (
                v[:n]
            )


            index = np.argmax(
                np.abs(
                    state_part
                )
            )


            v = (
                v
                / state_part[index]
            )


            chains.append([
                v
            ])


        total = (
            kernel.shape[1]
        )


        while (
            total
            < multiplicity
        ):

            extended = False


            for chain in chains:

                rhs = (
                    E
                    @ chain[-1]
                )


                candidate, *_ = (
                    np.linalg.lstsq(
                        K,
                        rhs,
                        rcond=None,
                    )
                )


                residual = (
                    K
                    @ candidate
                    - rhs
                )


                if (
                    np.linalg.norm(
                        residual
                    )
                    < 1e-7
                ):

                    candidate = (

                        candidate

                        - kernel
                        @ (
                            kernel.conj().T
                            @ candidate
                        )
                    )


                    chain.append(
                        candidate
                    )


                    total += 1

                    extended = True


                    if (
                        total
                        >= multiplicity
                    ):

                        break


            if not extended:

                raise RuntimeError(
                    "Rosenbrock chain construction failed."
                )


        for chain in chains:

            V_chain = (
                np.column_stack(
                    chain
                )
            )


            length = (
                len(
                    chain
                )
            )


            J_chain = (
                zero
                * np.eye(
                    length,
                    dtype=complex,
                )
            )


            for j in range(
                length - 1
            ):

                J_chain[
                    j,
                    j + 1
                ] = 1.0


            V_blocks.append(
                V_chain
            )


            J_blocks.append(
                J_chain
            )


    V_Z = np.real_if_close(
        np.column_stack(
            V_blocks
        )
    )


    J_Z = np.real_if_close(
        block_diag(
            *J_blocks
        )
    )


    V_Z = np.asarray(
        V_Z,
        dtype=float,
    )


    J_Z = np.asarray(
        J_Z,
        dtype=float,
    )


    return (
        V_Z[:n],
        V_Z[n:],
        J_Z,
        finite,
    )


V_x, V_u, J_Z, invariant_zeros = (
    rosenbrock_basis(
        A_full,
        B_full,
        C_zero,
        D_zero,
    )
)


V_x_pinv = (
    np.linalg.pinv(
        V_x
    )
)


# ============================================================
# 22. ESPAÇO TRANSVERSO
# ============================================================

Q_tangent = (
    orth(
        V_x
    )
)


Q_perp = (
    null_space(
        Q_tangent.T
    )
)


N_ZERO = (
    V_x.shape[1]
)


N_PERP = (
    Q_perp.shape[1]
)


# ============================================================
# 23. CONTROLADOR UNIFICADO TANGENCIAL-TRANSVERSAL (LQI)
#
# Transformacao:
#
#       dx = T_XI @ xi
#       xi = [eta ; z_perp]
#       T_XI = [V_x  Q_perp]
#
# Em seguida aumentamos o sistema com
#
#       zeta_theta = integral(theta dt)
#
# O seletor de theta em coordenadas xi NAO e assumido
# manualmente: ele e obtido da transformacao fisica.
# ============================================================

T_XI = np.hstack([
    V_x,
    Q_perp,
])

T_XI_COND = np.linalg.cond(T_XI)
T_XI_INV = np.linalg.inv(T_XI)

A_xi_c = T_XI_INV @ A_full @ T_XI
B_xi_c = T_XI_INV @ B_full
N_XI = A_xi_c.shape[0]

(
    A_xi_d,
    B_xi_d,
    _,
    _,
    _,
) = cont2discrete(
    (
        A_xi_c,
        B_xi_c,
        np.eye(N_XI),
        np.zeros((N_XI, 1)),
    ),
    T_SAMPLE,
    method="zoh",
)

Q_XI = block_diag(
    Q_ETA_WEIGHT,
    Q_ZPERP_WEIGHT,
)

# theta = C_THETA_X @ dx = C_THETA_XI @ xi
C_THETA_X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
C_THETA_XI = C_THETA_X @ T_XI

# Sistema discreto aumentado do LQI:
#
#   xi[k+1]   = A_xi_d xi[k] + B_xi_d duty[k]
#   zeta[k+1] = zeta[k] + Ts*C_THETA_XI*xi[k]
#
A_LQI = np.block([
    [
        A_xi_d,
        np.zeros((N_XI, 1)),
    ],
    [
        T_SAMPLE * C_THETA_XI,
        np.ones((1, 1)),
    ],
])

B_LQI = np.vstack([
    B_xi_d,
    np.zeros((1, 1)),
])

Q_LQI = block_diag(
    Q_XI,
    np.array([[Q_INTEGRAL_WEIGHT]]),
)

P_LQI = solve_discrete_are(
    A_LQI,
    B_LQI,
    Q_LQI,
    R_XI,
)

K_LQI = (
    np.linalg.inv(
        R_XI
        + B_LQI.T @ P_LQI @ B_LQI
    )
    @ (B_LQI.T @ P_LQI @ A_LQI)
)

K_XI = K_LQI[:, :N_XI]
K_I = K_LQI[:, N_XI:]

K_eta = K_XI[:, :N_ZERO]
K_perp = K_XI[:, N_ZERO:]

closed_loop_poles_lqi, _ = eig(
    A_LQI - B_LQI @ K_LQI
)

# ============================================================
# 24. MODELO REDUZIDO DISCRETO
# ============================================================

C_measure = np.array([
    [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
])


(
    A_red_d,
    B_red_d,
    _,
    _,
    _,
) = cont2discrete(

    (
        A_red,
        B_red,
        C_measure,

        np.zeros(
            (
                1,
                1,
            )
        ),
    ),

    T_SAMPLE,

    method="zoh",
)


# ============================================================
# 25. KALMAN
# ============================================================

P = solve_discrete_are(
    A_red_d.T,
    C_measure.T,
    Q_EST,
    R_EST,
)


S = (
    C_measure
    @ P
    @ C_measure.T
    + R_EST
)


K_kalman = (
    P
    @ C_measure.T
    @ np.linalg.inv(
        S
    )
)


# ============================================================
# 26. ESTADO AUMENTADO ESTIMADO
# ============================================================

def augmented_estimate(
    dx_hat_red,
    previous_duty,
):

    current_hat = (
        quasi_steady_current(
            dx_hat_red[2],
            previous_duty,
        )
    )


    return np.array([

        dx_hat_red[0],

        dx_hat_red[1],

        dx_hat_red[2],

        dx_hat_red[3],

        current_hat,
    ])


# ============================================================
# 27. CONTROLADOR LQI
# ============================================================

def control_law(
    dx_hat_red,
    previous_duty,
    theta_integral,
    theta_reference,
):

    dx_hat_aug = augmented_estimate(
        dx_hat_red,
        previous_duty,
    )

    # ------------------------------------------------------------------
    # BUG FIX: T_XI, K_eta, K_perp (and therefore eta_hat/z_hat) were all
    # derived from a linearization about x_eq_full = [0, pi, 0, 0, 0].
    # dx_hat_aug holds the ABSOLUTE state estimate (alpha ~ pi at the
    # top, not ~0; theta can also be far from 0 after the arm has spun
    # through several full turns during swing-up). Feeding that directly
    # through T_XI_INV multiplies an O(pi) (or O(many*2*pi)) quantity by
    # gains sized for small deviations, producing enormous bogus control
    # effort and saturating the actuator regardless of how close the
    # pendulum truly is to being balanced. Fix: re-center on the
    # equilibrium first.
    #
    # alpha: wrap so we always take the *shortest* angular deviation
    # from pi, regardless of how many turns alpha has accumulated.
    #
    # theta: there is no physical reason to unwind whatever number of
    # full turns the arm made during swing-up -- any fixed arm angle is
    # an equally good "home" position. We regulate deviation from
    # theta_reference (the arm angle captured at the moment the LQI took
    # over), not deviation from a hard-coded absolute zero. This avoids
    # the controller wasting/competing for torque authority trying to
    # unwind hundreds of degrees of accumulated rotation immediately
    # after a catch, which was observed to make the catch fail.
    # ------------------------------------------------------------------

    dx_centered = dx_hat_aug - x_eq_full
    dx_centered[0] = dx_hat_aug[0] - theta_reference
    dx_centered[1] = wrap_to_pi(dx_centered[1])

    xi_hat = T_XI_INV @ dx_centered

    eta_hat = xi_hat[:N_ZERO]
    z_hat = xi_hat[N_ZERO:]

    duty_eta = float(
        (-K_eta @ eta_hat).item()
    )

    duty_transverse = float(
        (-K_perp @ z_hat).item()
    )

    duty_integral = float(
        (-K_I * theta_integral).item()
    )

    duty_raw = (
        duty_eta
        + duty_transverse
        + duty_integral
    )

    duty = saturate_duty(duty_raw)

    return (
        duty,
        duty_raw,
        duty_eta,
        duty_transverse,
        duty_integral,
        eta_hat,
        z_hat,
        xi_hat,
        dx_hat_aug,
    )


# ============================================================
# 27B. ONE-SWING FINITE-HORIZON SYNTHESIS
# ============================================================

def compact_full_average_rhs_design(x, duty):
    """Fast averaged electromechanical Furuta model for offline shooting.

    State:
        x = [theta, alpha, theta_dot, alpha_dot, current]

    The mechanical equations are the compact symbolic equations generated by
    the same Lagrangian used by the main plant.  The electrical state is kept
    dynamically; only the 20 kHz switching ripple is averaged during design.

    The executed trajectory is still validated on multirate_plant_step().
    """

    theta_value, alpha_value, theta_dot_value, alpha_dot_value, current = x

    m_value = float(params[m])
    lr = float(params[Lr])
    lp = float(params[Lp])
    jr = float(params[Jr])
    jp = float(params[Jp])
    br_value = float(params[br])
    bp_value = float(params[bp])
    g_value = float(params[g])

    torque = motor_to_arm_torque(current)

    sin_alpha = np.sin(alpha_value)
    cos_alpha = np.cos(alpha_value)

    M11 = (
        jr
        + m_value
        * (
            lp**2 * sin_alpha**2
            + lr**2
        )
    )

    M12 = (
        m_value
        * lp
        * lr
        * cos_alpha
    )

    M22 = (
        jp
        + m_value * lp**2
    )

    h1 = (
        m_value
        * lp**2
        * alpha_dot_value
        * theta_dot_value
        * np.sin(2.0 * alpha_value)

        - m_value
        * lp
        * lr
        * alpha_dot_value**2
        * sin_alpha

        + br_value * theta_dot_value
        - torque
    )

    h2 = (
        -0.5
        * m_value
        * lp**2
        * theta_dot_value**2
        * np.sin(2.0 * alpha_value)

        + m_value
        * g_value
        * lp
        * sin_alpha

        + bp_value * alpha_dot_value
    )

    determinant = (
        M11 * M22
        - M12**2
    )

    theta_ddot_value = (
        -M22 * h1
        + M12 * h2
    ) / determinant

    alpha_ddot_value = (
        M12 * h1
        - M11 * h2
    ) / determinant

    current_dot = (
        V_DC * duty
        - R_MOTOR * current
        - K_E * GEAR_RATIO * theta_dot_value
    ) / L_MOTOR

    return np.array([
        theta_dot_value,
        alpha_dot_value,
        theta_ddot_value,
        alpha_ddot_value,
        current_dot,
    ])


def one_swing_edges(durations):
    return np.cumsum(
        np.asarray(
            durations,
            dtype=float,
        )
    )


def one_swing_duty(time_value, durations):
    durations = np.asarray(
        durations,
        dtype=float,
    )

    edges = one_swing_edges(
        durations
    )

    if time_value < 0.0:
        return 0.0

    if time_value >= edges[-1]:
        return 0.0

    segment = int(
        np.searchsorted(
            edges,
            time_value,
            side="right",
        )
    )

    segment = min(
        segment,
        len(ONE_SWING_SIGNS) - 1,
    )

    return float(
        ONE_SWING_SIGNS[segment]
    )


def simulate_one_swing_candidate(
    durations,
    x0_full,
    store=False,
):
    durations = np.asarray(
        durations,
        dtype=float,
    )

    duration_total = float(
        np.sum(durations)
    )

    n_step = max(
        1,
        int(
            np.ceil(
                duration_total
                / ONE_SWING_DESIGN_DT
            )
        ),
    )

    dt = (
        duration_total
        / n_step
    )

    x = np.asarray(
        x0_full,
        dtype=float,
    ).copy()

    if store:
        time_history = np.linspace(
            0.0,
            duration_total,
            n_step + 1,
        )

        state_history = np.zeros(
            (
                n_step + 1,
                N_FULL,
            )
        )

        duty_history_local = np.zeros(
            n_step
        )

        state_history[0] = x

    for k_local in range(n_step):
        t_local = k_local * dt

        duty = one_swing_duty(
            t_local,
            durations,
        )

        k1 = compact_full_average_rhs_design(
            x,
            duty,
        )

        k2 = compact_full_average_rhs_design(
            x + 0.5 * dt * k1,
            duty,
        )

        k3 = compact_full_average_rhs_design(
            x + 0.5 * dt * k2,
            duty,
        )

        k4 = compact_full_average_rhs_design(
            x + dt * k3,
            duty,
        )

        x = (
            x
            + dt
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
            / 6.0
        )

        if not np.all(
            np.isfinite(x)
        ):
            return None

        if store:
            state_history[k_local + 1] = x
            duty_history_local[k_local] = duty

    if not store:
        return x

    return {
        "duration": duration_total,
        "durations": durations.copy(),
        "edges": one_swing_edges(durations),
        "time": time_history,
        "state": state_history,
        "duty": duty_history_local,
    }


def one_swing_objective(
    durations,
    x0_full,
):
    plan = simulate_one_swing_candidate(
        durations,
        x0_full,
        store=True,
    )

    if plan is None:
        return 1.0e12

    xf = plan["state"][-1]

    alpha_error_final = wrap_to_pi(
        xf[1] - np.pi
    )

    cost = (
        (
            alpha_error_final
            / ONE_SWING_TARGET_ANGLE
        )**2

        + (
            xf[3]
            / ONE_SWING_TARGET_ALPHA_RATE
        )**2

        + 2.0
        * (
            xf[2]
            / ONE_SWING_TARGET_THETA_RATE
        )**2

        + 0.35
        * (
            xf[0]
            / ONE_SWING_HANDOFF_THETA_LIMIT
        )**2
    )

    # Hard path requirement: the arm must never cross +/- 360 degrees.
    theta_abs_max = float(np.max(np.abs(plan["state"][:, 0])))
    if theta_abs_max > ONE_SWING_THETA_LIMIT:
        violation = (theta_abs_max - ONE_SWING_THETA_LIMIT) / np.deg2rad(5.0)
        cost += 1.0e5 * (1.0 + violation**2)

    # Leave geometric margin for the LQI to return the arm to theta=0.
    if abs(xf[0]) > ONE_SWING_HANDOFF_THETA_LIMIT:
        violation = (abs(xf[0]) - ONE_SWING_HANDOFF_THETA_LIMIT) / np.deg2rad(10.0)
        cost += 2.0e3 * (1.0 + violation**2)

    # Keep the trajectory in a genuine one-swing family rather than allowing
    # complete pendulum revolutions.  A moderate backswing is allowed because
    # it is exactly what supplies the kinetic energy for the upward stroke.
    alpha_unwrapped = np.unwrap(
        plan["state"][:, 1]
    )

    alpha_max = float(
        np.max(alpha_unwrapped)
    )

    alpha_min = float(
        np.min(alpha_unwrapped)
    )

    if alpha_max > np.deg2rad(225.0):
        cost += (
            30.0
            * (
                (
                    alpha_max
                    - np.deg2rad(225.0)
                )
                / np.deg2rad(15.0)
            )**2
        )

    if alpha_min < np.deg2rad(-105.0):
        cost += (
            30.0
            * (
                (
                    np.deg2rad(-105.0)
                    - alpha_min
                )
                / np.deg2rad(15.0)
            )**2
        )

    return float(cost)


def design_one_swing(x0_full):
    bounds = [ONE_SWING_DURATION_BOUNDS for _ in ONE_SWING_SIGNS]

    best_result = None
    best_plan = None

    for seed in ONE_SWING_SEEDS:
        result = differential_evolution(
            lambda durations: one_swing_objective(durations, x0_full),
            bounds=bounds,
            maxiter=ONE_SWING_MAXITER,
            popsize=ONE_SWING_POPSIZE,
            seed=seed,
            polish=True,
            tol=1.0e-5,
            updating="immediate",
            workers=1,
            x0=ONE_SWING_NOMINAL_DURATIONS,
        )
        plan = simulate_one_swing_candidate(result.x, x0_full, store=True)
        if plan is None:
            continue
        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_plan = plan

        xf = plan["state"][-1]
        ae = wrap_to_pi(xf[1] - np.pi)
        feasible = (
            abs(ae) < SWING_UP_SWITCH_ANGLE
            and abs(xf[3]) < SWING_UP_SWITCH_ALPHA_RATE
            and abs(xf[2]) < SWING_UP_SWITCH_THETA_RATE
            and abs(xf[0]) < ONE_SWING_HANDOFF_THETA_LIMIT
            and np.max(np.abs(plan["state"][:, 0])) <= ONE_SWING_THETA_LIMIT
        )
        if feasible:
            best_result = result
            best_plan = plan
            break

    if best_plan is None:
        raise RuntimeError("One-swing optimizer produced no finite candidate.")

    best_plan["optimizer_cost"] = float(best_result.fun)
    best_plan["optimizer_success"] = bool(best_result.success)
    return best_plan

def build_one_swing_plan(x0_full):
    if ONE_SWING_REDESIGN:
        plan = design_one_swing(
            x0_full
        )
    else:
        plan = simulate_one_swing_candidate(
            ONE_SWING_NOMINAL_DURATIONS,
            x0_full,
            store=True,
        )

        plan["optimizer_cost"] = one_swing_objective(
            ONE_SWING_NOMINAL_DURATIONS,
            x0_full,
        )

        plan["optimizer_success"] = True

    xf = plan["state"][-1]

    plan["terminal_alpha_error"] = wrap_to_pi(
        xf[1] - np.pi
    )

    plan["terminal_alpha_rate"] = float(
        xf[3]
    )

    plan["terminal_theta_rate"] = float(
        xf[2]
    )

    plan["terminal_theta"] = float(xf[0])
    plan["max_abs_theta"] = float(np.max(np.abs(plan["state"][:, 0])))

    plan["inside_capture_nominal"] = (
        abs(
            plan["terminal_alpha_error"]
        ) < SWING_UP_SWITCH_ANGLE

        and abs(
            plan["terminal_alpha_rate"]
        ) < SWING_UP_SWITCH_ALPHA_RATE

        and abs(
            plan["terminal_theta_rate"]
        ) < SWING_UP_SWITCH_THETA_RATE

        and abs(plan["terminal_theta"]) < ONE_SWING_HANDOFF_THETA_LIMIT

        and plan["max_abs_theta"] <= ONE_SWING_THETA_LIMIT
    )

    return plan


def energy_swing_up_fallback(
    x_mech,
):
    theta_value, alpha_value, theta_dot_value, alpha_dot_value = x_mech

    energy = (
        0.5
        * PENDULUM_INERTIA
        * alpha_dot_value**2

        - PENDULUM_ENERGY_TARGET
        * np.cos(alpha_value)
    )

    energy_error = (
        energy
        - PENDULUM_ENERGY_TARGET
    )

    direction = np.sign(
        alpha_dot_value
        * np.cos(alpha_value)
    )

    if direction == 0.0:
        direction = 1.0

    if energy_error < -SWING_UP_ENERGY_DEADBAND:
        duty_raw = -direction

    elif energy_error > SWING_UP_ENERGY_DEADBAND:
        duty_raw = direction

    else:
        duty_raw = 0.0

    return (
        saturate_duty(duty_raw),
        float(duty_raw),
        float(energy),
    )


def swing_up_control_law(
    time_value,
    x_mech,
    previous_duty,
    one_swing_plan,
):
    """Execute the precomputed one-swing plan.

    If the plant was changed and the plan ends outside the capture region, an
    energy-pump fallback can recover rather than leaving the pendulum open-loop.
    """

    dx_hat_aug = augmented_estimate(
        x_mech,
        previous_duty,
    )

    if (
        ONE_SWING_ENABLE
        and time_value < one_swing_plan["duration"]
    ):
        duty_raw = one_swing_duty(
            time_value,
            one_swing_plan["durations"],
        )

        duty = saturate_duty(
            duty_raw
        )

        energy = (
            0.5
            * PENDULUM_INERTIA
            * x_mech[3]**2

            - PENDULUM_ENERGY_TARGET
            * np.cos(x_mech[1])
        )

    elif ONE_SWING_FALLBACK_TO_ENERGY:
        (
            duty,
            duty_raw,
            energy,
        ) = energy_swing_up_fallback(
            x_mech
        )

    else:
        duty = 0.0
        duty_raw = 0.0

        energy = (
            0.5
            * PENDULUM_INERTIA
            * x_mech[3]**2

            - PENDULUM_ENERGY_TARGET
            * np.cos(x_mech[1])
        )

    return (
        duty,
        duty_raw,
        dx_hat_aug,
        energy,
        np.nan,
        np.nan,
    )


# ============================================================
# 28. KALMAN PREDICT / CORRECT
# ============================================================

def kalman_step(
    dx_hat,
    duty,
    measurement_next,
):

    # ------------------------------------------------------------------
    # BUG FIX: A_red_d/B_red_d come from a linearization of reduced_rhs
    # about x_eq_red = [0, pi, 0, 0]. Exactly like control_law, this
    # predictor is only valid acting on the DEVIATION from that
    # equilibrium, not on the absolute state. dx_hat's alpha component is
    # ~pi (or many turns away from it during/just after swing-up), so it
    # must be centered (with wraparound, since alpha can accumulate many
    # full turns) before being pushed through the linear predictor, and
    # shifted back afterwards.
    # ------------------------------------------------------------------

    dx_dev = dx_hat - x_eq_red
    dx_dev[1] = wrap_to_pi(dx_dev[1])

    prediction_dev = (
        A_red_d
        @ dx_dev

        + B_red_d[:, 0]
        * duty
    )


    measurement_dev = (
        measurement_next
        - x_eq_red[0]
    )

    innovation = (
        measurement_dev

        - float(
            (
                C_measure
                @ prediction_dev
            ).item()
        )
    )


    corrected_dev = (
        prediction_dev

        + K_kalman[:, 0]
        * innovation
    )

    corrected = corrected_dev + x_eq_red


    return (
        corrected,
        innovation,
    )


# ============================================================
# 29. CONDIÇÕES INICIAIS
# ============================================================

x_initial = np.array([

    0.0,

    0.0,

    0.0,

    np.deg2rad(
        5.0
    ),

    0.0,
])


dx_hat_initial = x_initial[:4].copy()


# ============================================================
# 29B. ONE-SWING PLAN
# ============================================================

if ONE_SWING_ENABLE:
    print("\n========================================")
    print("ONE-SWING TRAJECTORY")
    print("========================================")

    ONE_SWING_PLAN = build_one_swing_plan(
        x_initial
    )

    print(
        "signs =",
        ONE_SWING_SIGNS,
    )

    print(
        "durations [s] =",
        ONE_SWING_PLAN["durations"],
    )

    print(
        "total duration [s] =",
        ONE_SWING_PLAN["duration"],
    )

    print(
        "objective =",
        ONE_SWING_PLAN["optimizer_cost"],
    )

    print(
        "nominal terminal alpha error [deg] =",
        np.rad2deg(
            ONE_SWING_PLAN["terminal_alpha_error"]
        ),
    )

    print(
        "nominal terminal alpha_dot [deg/s] =",
        np.rad2deg(
            ONE_SWING_PLAN["terminal_alpha_rate"]
        ),
    )

    print(
        "nominal terminal theta_dot [deg/s] =",
        np.rad2deg(
            ONE_SWING_PLAN["terminal_theta_rate"]
        ),
    )

    print(
        "nominal terminal state inside LQI capture =",
        ONE_SWING_PLAN["inside_capture_nominal"],
    )

    if not ONE_SWING_PLAN["inside_capture_nominal"]:
        raise RuntimeError(
            "One-swing synthesis did not find a feasible trajectory satisfying "
            "a capture state with arm margin, |theta(t)|<=360 deg, and the LQI "
            "capture constraints. Inspect the printed terminal metrics and "
            "increase the trajectory degrees of freedom or redesign settings."
        )

else:
    ONE_SWING_PLAN = {
        "duration": 0.0,
        "durations": np.zeros(
            len(ONE_SWING_SIGNS)
        ),
        "inside_capture_nominal": False,
    }


# ============================================================
# 30. HISTÓRICOS
# ============================================================

N_SAMPLE = int(
    round(
        SIMULATION_TIME
        / T_SAMPLE
    )
)


times = (
    np.arange(
        N_SAMPLE + 1
    )
    * T_SAMPLE
)


x_history = np.zeros(
    (
        N_SAMPLE + 1,
        N_FULL,
    )
)


dx_hat_red_history = np.zeros(
    (
        N_SAMPLE + 1,
        N_RED,
    )
)


dx_hat_aug_history = np.zeros(
    (
        N_SAMPLE + 1,
        N_FULL,
    )
)


duty_history = np.zeros(
    N_SAMPLE
)


lqr_active_history = np.zeros(
    N_SAMPLE,
    dtype=bool,
)


duty_raw_history = np.zeros(
    N_SAMPLE
)


duty_eta_history = np.zeros(
    N_SAMPLE
)


duty_transverse_history = np.zeros(
    N_SAMPLE
)


duty_integral_history = np.zeros(
    N_SAMPLE
)


theta_integral_history = np.zeros(
    N_SAMPLE + 1
)


innovation_history = np.zeros(
    N_SAMPLE
)


average_current_history = np.zeros(
    N_SAMPLE
)


average_torque_history = np.zeros(
    N_SAMPLE
)


x_history[0] = (
    x_initial
)


dx_hat_red_history[0] = (
    dx_hat_initial
)


previous_duty = 0.0
theta_integral = 0.0
theta_reference = 0.0
lqr_active = False


# ============================================================
# 31. INFO
# ============================================================

print("\n========================================")
print("FURUTA — MULTIRATE PWM PIPELINE")
print("========================================")


print(
    f"\nControl frequency = "
    f"{F_SAMPLE:.1f} Hz"
)


print(
    f"Control period = "
    f"{T_SAMPLE*1000:.3f} ms"
)


print(
    f"PWM frequency = "
    f"{F_PWM:.1f} Hz"
)


print(
    f"PWM period = "
    f"{T_PWM*1e6:.3f} us"
)


print(
    f"PWM cycles / control interval = "
    f"{PWM_CYCLES_PER_CONTROL}"
)


print(
    f"Electrical time constant = "
    f"{TAU_ELECTRICAL*1000:.3f} ms"
)


print(
    f"Ts / tau_e = "
    f"{T_SAMPLE/TAU_ELECTRICAL:.3f}"
)


print("\nInvariant zeros:")
print(
    invariant_zeros
)


print("\nV_x =")
print(
    V_x
)


print("\nV_u (referencia, nao usado no controlador unificado) =")
print(
    V_u
)


print("\nJ_Z =")
print(
    J_Z
)


print(
    f"\ncond(T_XI) = {T_XI_COND:.4f}"
    f"  (baixo => decomposicao eta/z_perp bem condicionada)"
)


print("\nQ_LQI = blockdiag(Q_eta, Q_zperp, q_integral) =")
print(Q_LQI)

print("\nC_THETA_XI =")
print(C_THETA_XI)

print("\nK_LQI = [K_eta  K_perp  K_I] =")
print(K_LQI)

print("\nK_eta =")
print(K_eta)

print("\nK_perp =")
print(K_perp)

print("\nK_I =")
print(K_I)

print("\nPolos de malha fechada LQI (discretos, |z|<1):")
print(closed_loop_poles_lqi)
print("  |polos| =", np.abs(closed_loop_poles_lqi))


print("\nK_kalman =")
print(
    K_kalman
)


# ============================================================
# 32. SIMULAÇÃO
# ============================================================

print("\n========================================")
print("RUNNING SIMULATION")
print("========================================")


for k in range(
    N_SAMPLE
):

    x_k = (
        x_history[k]
    )


    dx_hat_k = (
        dx_hat_red_history[k]
    )


    # --------------------------------------------------------
    # Controlador
    # --------------------------------------------------------

    swing_up_state = x_k[:4]
    dx_hat_aug_candidate = augmented_estimate(
        swing_up_state,
        previous_duty,
    )
    alpha_error = wrap_to_pi(
        dx_hat_aug_candidate[1] - np.pi
    )

    if (
        not lqr_active
        and times[k] >= ONE_SWING_PLAN["duration"]
        and abs(alpha_error) < SWING_UP_SWITCH_ANGLE
        and abs(dx_hat_aug_candidate[3]) < SWING_UP_SWITCH_ALPHA_RATE
        and abs(dx_hat_aug_candidate[2]) < SWING_UP_SWITCH_THETA_RATE
        and abs(dx_hat_aug_candidate[0]) < ONE_SWING_HANDOFF_THETA_LIMIT
    ):
        lqr_active = True
        theta_integral = 0.0
        # The planner only needs to enter the LQI capture region while
        # preserving enough angular margin. From the handoff onward, the
        # LQI regulates the absolute physical equilibrium theta = 0.
        theta_reference = 0.0

        # BUG FIX (bumpless transfer): the reduced Kalman filter's linear
        # model is only valid near the top equilibrium. During the
        # violent, large-angle swing-up it tracks theta/theta_dot
        # reasonably (they are directly measured / well-behaved) but
        # alpha/alpha_dot (never directly measured, purely propagated
        # through a small-signal model) drift completely away from
        # truth -- verified directly: at one handoff instant the true
        # alpha error was -25 deg / +141 deg/s while the filter's
        # estimate said -80 deg / -616 deg/s. Handing that garbage
        # estimate to the LQR guarantees an immediate loss of the catch
        # regardless of how good the controller's true capture basin is.
        # Swing-up already uses the true mechanical state directly (see
        # swing_up_state below), so at the transition we simply
        # re-initialize the estimator with that same true state -- a
        # standard bumpless-transfer reset -- and let the Kalman filter
        # track from a correct starting point once we are actually near
        # the equilibrium where its linear model applies.
        dx_hat_k = swing_up_state.copy()

    elif (
        lqr_active
        and abs(alpha_error) > SWING_UP_REENGAGE_ANGLE
    ):
        # Lost the catch: fall back to the swing-up pump instead of
        # letting the linear controller keep fighting (and failing) far
        # outside its region of validity.
        lqr_active = False
        theta_integral = 0.0

    if lqr_active:
        (
            duty,
            duty_raw,
            duty_eta,
            duty_transverse,
            duty_integral,
            _,
            _,
            _,
            dx_hat_aug,
        ) = control_law(
            dx_hat_k,
            previous_duty,
            theta_integral,
            theta_reference,
        )
    else:
        (
            duty,
            duty_raw,
            dx_hat_aug,
            _,
            _,
            _,
        ) = swing_up_control_law(
            times[k],
            swing_up_state,
            previous_duty,
            ONE_SWING_PLAN,
        )
        duty_eta = 0.0
        duty_transverse = 0.0
        duty_integral = 0.0

    lqr_active_history[k] = lqr_active


    duty_history[k] = (
        duty
    )


    duty_raw_history[k] = (
        duty_raw
    )


    duty_eta_history[k] = (
        duty_eta
    )


    duty_transverse_history[k] = (
        duty_transverse
    )


    duty_integral_history[k] = (
        duty_integral
    )


    dx_hat_aug_history[k] = (
        dx_hat_aug
    )


    # Hard mechanical travel constraint required by the one-swing specification.
    if abs(x_k[0]) > ONE_SWING_THETA_LIMIT + 1e-9:
        raise RuntimeError(
            f"Arm travel constraint violated at t={times[k]:.6f} s: "
            f"theta={np.rad2deg(x_k[0]):.3f} deg."
        )

    # --------------------------------------------------------
    # Planta multirate
    # --------------------------------------------------------

    plant_result = (
        multirate_plant_step(
            x_k,
            duty,
            store_pwm=True,
        )
    )


    x_next = (
        plant_result[
            "x_next"
        ]
    )


    average_current_history[k] = (
        plant_result[
            "current_average"
        ]
    )


    average_torque_history[k] = (
        plant_result[
            "torque_average"
        ]
    )


    x_history[
        k + 1
    ] = (
        x_next
    )

    if abs(x_next[0]) > ONE_SWING_THETA_LIMIT + 1e-9:
        raise RuntimeError(
            f"Arm travel limit violated at t={times[k+1]:.4f} s: "
            f"theta={np.rad2deg(x_next[0]):.3f} deg"
        )


    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    measurement_next = (
        x_next[0]
    )


    # --------------------------------------------------------
    # Kalman
    # --------------------------------------------------------

    (
        dx_hat_next,
        innovation,

    ) = kalman_step(
        dx_hat_k,
        duty,
        measurement_next,
    )


    dx_hat_red_history[
        k + 1
    ] = (
        dx_hat_next
    )


    innovation_history[k] = (
        innovation
    )


    # --------------------------------------------------------
    # Integrador de theta + anti-windup condicional
    #
    # Integramos quando o atuador nao esta saturado OU quando
    # o erro de theta tenderia a empurrar o comando para fora
    # da saturacao. Isso evita windup mantendo capacidade de
    # recuperacao durante saturacao transitoria.
    # --------------------------------------------------------

    theta_hat = float(dx_hat_k[0]) - theta_reference

    not_saturated = abs(duty_raw) <= MAX_DUTY

    # Sensibilidade do duty ao integrador: -K_I.
    # Se duty_raw > +limit, queremos Delta duty < 0.
    # Se duty_raw < -limit, queremos Delta duty > 0.
    integral_would_desaturate = (
        (duty_raw > MAX_DUTY and float((-K_I * theta_hat).item()) < 0.0)
        or
        (duty_raw < -MAX_DUTY and float((-K_I * theta_hat).item()) > 0.0)
    )

    if not_saturated or integral_would_desaturate:
        theta_integral += T_SAMPLE * theta_hat

    theta_integral = float(np.clip(
        theta_integral,
        -THETA_INTEGRAL_LIMIT,
        THETA_INTEGRAL_LIMIT,
    ))

    theta_integral_history[k + 1] = theta_integral


    previous_duty = (
        duty
    )


    if (
        k % 500
        == 0
    ):

        alpha_error = (
            wrap_to_pi(
                x_k[1]
                - np.pi
            )
        )


        print(
            f"\r"
            f"{k:5d}/{N_SAMPLE}"
            f"  duty={duty:+.4f}"
            f"  alpha_err="
            f"{np.rad2deg(alpha_error):+.5f} deg"
            f"  i={x_k[4]:+.4f} A",
            end="",
            flush=True,
        )


print(
    "\nSimulation finished."
)


# ============================================================
# 33. ESTIMATIVA FINAL
# ============================================================

dx_hat_aug_history[-1] = (
    augmented_estimate(
        dx_hat_red_history[-1],
        previous_duty,
    )
)


# ============================================================
# 34. DIAGNÓSTICOS
# ============================================================

dx_true_history = (
    x_history
    - x_eq_full
)


for k in range(
    N_SAMPLE + 1
):

    dx_true_history[
        k,
        1
    ] = (
        wrap_to_pi(
            x_history[k, 1]
            - np.pi
        )
    )


mechanical_error = (
    x_history[:, :4]
    - dx_hat_red_history
)

# NOTE: column 1 is alpha. Both x_history and dx_hat_red_history store
# the *absolute* (unwrapped) mechanical angle here, so a plain
# subtraction is the right comparison -- but alpha can differ by whole
# multiples of 2*pi (accumulated arm/pendulum rotations) while
# physically representing the same angle, so the raw difference must be
# wrapped to get the true, minimal angular error.
mechanical_error[:, 1] = wrap_to_pi(
    mechanical_error[:, 1]
)


mechanical_error_norm = (
    np.linalg.norm(
        mechanical_error,
        axis=1,
    )
)


current_true = (
    x_history[:, 4]
)


current_hat = (
    dx_hat_aug_history[:, 4]
)


current_error = (
    current_true
    - current_hat
)


# --------------------------------------------------------
# Corrente "verdadeira" para fins do diagnostico transverso
#
# x_history[:, 4] e amostrada sempre na MESMA FASE do PWM:
# o fim do ultimo trecho OFF de cada intervalo de controle.
# Com Ts/tau_e = 1 essa amostra carrega o VALE do ripple de
# comutacao, nao a media do ciclo.
#
# A reconstrucao do observador (dx_hat_aug[:, 4], via
# quasi_steady_current) e, por construcao, a MEDIA de ciclo
# (equivalente a average_current_history). Comparar o vale
# do ripple contra essa media introduz um vies espurio e
# constante em z_true que nao tem relacao com o desempenho
# do controlador nem com a linearizacao de Rosenbrock — e
# so um descasamento de definicao entre duas grandezas
# fisicamente diferentes.
#
# Para a projecao transversa, usamos a corrente MEDIA de
# cada intervalo (mesma base da reconstrucao do observador),
# alinhando amostra-a-amostra: average_current_history[k]
# corresponde ao intervalo [x_history[k], x_history[k+1]).
# --------------------------------------------------------

current_true_cycle_avg = np.concatenate([
    average_current_history,
    average_current_history[-1:],
])


dx_true_history_transverse = (
    dx_true_history.copy()
)


dx_true_history_transverse[:, 4] = (
    current_true_cycle_avg
    - x_eq_full[4]
)


z_true = (
    Q_perp.T
    @ dx_true_history_transverse.T
).T


z_hat = (
    Q_perp.T
    @ dx_hat_aug_history.T
).T


z_true_norm = (
    np.linalg.norm(
        z_true,
        axis=1,
    )
)


z_hat_norm = (
    np.linalg.norm(
        z_hat,
        axis=1,
    )
)


alpha_error_deg = (
    np.rad2deg(
        dx_true_history[:, 1]
    )
)


alpha_dot_deg_s = (
    np.rad2deg(
        x_history[:, 3]
    )
)


theta_deg = (
    np.rad2deg(
        x_history[:, 0]
    )
)


theta_dot_deg_s = (
    np.rad2deg(
        x_history[:, 2]
    )
)


saturation_fraction = (
    np.mean(
        np.abs(
            duty_raw_history
        )
        > MAX_DUTY
    )
)


# ============================================================
# 35. PWM DEBUG WINDOW
#
# Re-simula apenas o primeiro intervalo para visualizar ON/OFF.
# ============================================================

pwm_debug = (
    multirate_plant_step(
        x_history[0],
        duty_history[0],
        store_pwm=True,
    )
)


pwm_debug_time = (
    pwm_debug[
        "pwm_time"
    ]
)


pwm_debug_current = (
    pwm_debug[
        "pwm_current"
    ]
)


pwm_debug_voltage = (
    pwm_debug[
        "pwm_voltage"
    ]
)


pwm_debug_torque = (
    GEAR_EFFICIENCY
    * GEAR_RATIO
    * K_T
    * pwm_debug_current
)


# ============================================================
# 36. RESULTADOS
# ============================================================

print("\n========================================")
print("FINAL RESULTS")
print("========================================")


print(
    "\nInitial observer error =",
    mechanical_error_norm[0],
)


print(
    "Final observer error =",
    mechanical_error_norm[-1],
)


print(
    "\nInitial transverse norm =",
    z_true_norm[0],
)


print(
    "Final transverse norm =",
    z_true_norm[-1],
)


print(
    "\nFinal alpha error [deg] =",
    alpha_error_deg[-1],
)


print(
    "Final alpha_dot [deg/s] =",
    alpha_dot_deg_s[-1],
)


print(
    "\nFinal theta [deg] =",
    theta_deg[-1],
)


print(
    "Final theta_dot [deg/s] =",
    theta_dot_deg_s[-1],
)


print(
    "\nFinal true current [A] =",
    current_true[-1],
)


print(
    "Final reconstructed current [A] =",
    current_hat[-1],
)


print(
    "Final current reconstruction error [A] =",
    current_error[-1],
)


print(
    "\nMaximum |current reconstruction error| [A] =",
    np.max(
        np.abs(
            current_error
        )
    ),
)


print(
    "\nMaximum |duty| =",
    np.max(
        np.abs(
            duty_history
        )
    ),
)


print(
    "Saturation fraction =",
    saturation_fraction,
)


handoff_indices = np.flatnonzero(
    lqr_active_history
)

print(
    "LQR handoff time [s] =",
    times[handoff_indices[0]] if handoff_indices.size else "not reached",
)


print(
    "\nMaximum |average torque| [Nm] =",
    np.max(
        np.abs(
            average_torque_history
        )
    ),
)


# ============================================================
# 37. PLOT — KALMAN
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.semilogy(
    times,

    np.maximum(
        mechanical_error_norm,
        1e-16,
    ),
)


ax.set_xlabel(
    "time [s]"
)

ax.set_ylabel(
    r"$\|x_m-\hat{x}_m\|$"
)

ax.set_title(
    "Reduced Kalman estimator"
)

ax.grid(
    True,
    which="both",
)


fig.tight_layout()

fig.savefig(
    "furuta_kalman_error.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 38. PLOT — TRANSVERSAL
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.semilogy(
    times,

    np.maximum(
        z_true_norm,
        1e-16,
    ),

    label="true",
)


ax.semilogy(
    times,

    np.maximum(
        z_hat_norm,
        1e-16,
    ),

    "--",

    label="estimated",
)


ax.set_xlabel(
    "time [s]"
)

ax.set_ylabel(
    r"$\|z_\perp\|$"
)

ax.set_title(
    "Transverse stabilization"
)

ax.grid(
    True,
    which="both",
)

ax.legend()


fig.tight_layout()

fig.savefig(
    "furuta_transverse.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 39. PLOT — ALPHA
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.plot(
    times,
    alpha_error_deg,
)


ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
)


ax.set_xlabel(
    "time [s]"
)

ax.set_ylabel(
    r"$\alpha-\pi$ [deg]"
)

ax.set_title(
    "Pendulum stabilization"
)

ax.grid(True)


fig.tight_layout()

fig.savefig(
    "furuta_alpha.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 40. PLOT — DUTY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.step(
    times[:-1],
    duty_history,
    where="post",
    label="applied",
)


ax.step(
    times[:-1],
    duty_eta_history,
    where="post",
    linestyle=":",
    label="tangential (-K_eta eta)",
)


ax.step(
    times[:-1],
    duty_transverse_history,
    where="post",
    linestyle="--",
    label="transverse feedback",
)

ax.step(
    times[:-1],
    duty_integral_history,
    where="post",
    linestyle="-.",
    label="integral feedback",
)


ax.set_xlabel(
    "time [s]"
)

ax.set_ylabel(
    "duty"
)

ax.set_title(
    "Control command"
)

ax.grid(True)

ax.legend()


fig.tight_layout()

fig.savefig(
    "furuta_duty.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 41. PLOT — CORRENTE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.plot(
    times,
    current_true,
    label="true end-of-sample current",
)


ax.plot(
    times,
    current_hat,
    "--",
    label="quasi-steady reconstruction",
)


ax.plot(
    times[:-1],
    average_current_history,
    ":",
    label="PWM-average current",
)


ax.set_xlabel(
    "time [s]"
)

ax.set_ylabel(
    "current [A]"
)

ax.set_title(
    "Motor current"
)

ax.grid(True)

ax.legend()


fig.tight_layout()

fig.savefig(
    "furuta_current.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 42. PLOT — TORQUE MÉDIO
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.plot(
    times[:-1],
    average_torque_history,
)


ax.set_xlabel(
    "time [s]"
)

ax.set_ylabel(
    "average torque [Nm]"
)

ax.set_title(
    "Average motor torque per control interval"
)

ax.grid(True)


fig.tight_layout()

fig.savefig(
    "furuta_average_torque.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 43. PLOT — PWM EXPLÍCITO
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 9),
    sharex=True,
)


axes[0].step(
    pwm_debug_time * 1000.0,
    pwm_debug_voltage,
    where="pre",
)


axes[0].set_ylabel(
    "voltage [V]"
)

axes[0].set_title(
    "Explicit PWM — one control interval"
)

axes[0].grid(True)


axes[1].plot(
    pwm_debug_time * 1000.0,
    pwm_debug_current,
)


axes[1].set_ylabel(
    "current [A]"
)

axes[1].grid(True)


axes[2].plot(
    pwm_debug_time * 1000.0,
    pwm_debug_torque,
)


axes[2].set_ylabel(
    "torque [Nm]"
)

axes[2].set_xlabel(
    "time within control interval [ms]"
)

axes[2].grid(True)


fig.tight_layout()

fig.savefig(
    "furuta_pwm_interval.png",
    dpi=180,
)

plt.close(fig)


# ============================================================
# 44. PLOT — ESTADOS
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9),
)


axes[0, 0].plot(
    times,
    theta_deg,
)

axes[0, 0].set_title(
    r"$\theta$"
)

axes[0, 0].set_ylabel(
    "deg"
)

axes[0, 0].grid(True)


axes[0, 1].plot(
    times,
    alpha_error_deg,
)

axes[0, 1].set_title(
    r"$\alpha-\pi$"
)

axes[0, 1].set_ylabel(
    "deg"
)

axes[0, 1].grid(True)


axes[1, 0].plot(
    times,
    theta_dot_deg_s,
)

axes[1, 0].set_title(
    r"$\dot\theta$"
)

axes[1, 0].set_ylabel(
    "deg/s"
)

axes[1, 0].set_xlabel(
    "time [s]"
)

axes[1, 0].grid(True)


axes[1, 1].plot(
    times,
    alpha_dot_deg_s,
)

axes[1, 1].set_title(
    r"$\dot\alpha$"
)

axes[1, 1].set_ylabel(
    "deg/s"
)

axes[1, 1].set_xlabel(
    "time [s]"
)

axes[1, 1].grid(True)


fig.tight_layout()

fig.savefig(
    "furuta_states.png",
    dpi=150,
)

plt.close(fig)


# ============================================================
# 44B. PLOT — ESTADO INTEGRAL DO LQI
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    times,
    theta_integral_history,
)

ax.set_xlabel("time [s]")
ax.set_ylabel(r"$\zeta_\theta = \int \hat\theta dt$ [rad s]")
ax.set_title("LQI integral state")
ax.grid(True)

fig.tight_layout()
fig.savefig(
    "furuta_lqi_integral.png",
    dpi=150,
)
plt.close(fig)


# ============================================================
# 45. ARQUIVOS
# ============================================================

print("\n========================================")
print("GENERATED FILES")
print("========================================")


for filename in [

    "furuta_kalman_error.png",

    "furuta_transverse.png",

    "furuta_alpha.png",

    "furuta_duty.png",

    "furuta_current.png",

    "furuta_average_torque.png",

    "furuta_pwm_interval.png",

    "furuta_states.png",

    "furuta_lqi_integral.png",
    "furuta_one_swing_plan.png",

]:

    print(
        " ",
        filename,
    )

# ============================================================
# ONE-SWING NOMINAL PLAN PLOT
# ============================================================

if ONE_SWING_ENABLE:
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    plan_time = ONE_SWING_PLAN["time"]
    plan_state = ONE_SWING_PLAN["state"]

    axes[0].plot(
        plan_time,
        np.rad2deg(
            plan_state[:, 1]
        ),
    )

    axes[0].axhline(
        180.0,
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_ylabel(
        "alpha [deg]"
    )

    axes[1].plot(
        plan_time,
        np.rad2deg(
            plan_state[:, 3]
        ),
    )

    axes[1].set_ylabel(
        "alpha_dot [deg/s]"
    )

    axes[2].plot(
        plan_time,
        np.rad2deg(
            plan_state[:, 2]
        ),
    )

    axes[2].set_ylabel(
        "theta_dot [deg/s]"
    )

    axes[3].step(
        plan_time[:-1],
        ONE_SWING_PLAN["duty"],
        where="post",
    )

    axes[3].set_ylabel(
        "duty"
    )

    axes[3].set_xlabel(
        "time [s]"
    )

    for axis in axes:
        axis.grid(True)

    fig.suptitle(
        "Synthesized one-swing trajectory"
    )

    fig.tight_layout()

    fig.savefig(
        "furuta_one_swing_plan.png",
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# VIDEO ANIMATION
# ============================================================

from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import shutil


SAVE_VIDEO = False

VIDEO_FILENAME = "furuta_simulation.mp4"

VIDEO_FPS = 30

VIDEO_DURATION = 12.0


if SAVE_VIDEO:

    print("\n========================================")
    print("GENERATING VIDEO")
    print("========================================")


    # --------------------------------------------------------
    # Number of animation frames
    # --------------------------------------------------------

    N_VIDEO_FRAMES = int(
        VIDEO_FPS
        * VIDEO_DURATION
    )


    frame_indices = np.linspace(
        0,
        N_SAMPLE,
        N_VIDEO_FRAMES,
        dtype=int,
    )


    # --------------------------------------------------------
    # Physical dimensions
    # --------------------------------------------------------

    Lr_value = float(
        params[Lr]
    )

    Lp_value = float(
        params[Lp]
    )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )


    ax = fig.add_subplot(
        111,
        projection="3d",
    )


    axis_limit = (
        Lr_value
        + Lp_value
    ) * 1.25


    ax.set_xlim(
        -axis_limit,
        axis_limit,
    )

    ax.set_ylim(
        -axis_limit,
        axis_limit,
    )

    ax.set_zlim(
        -Lp_value * 1.2,
        Lp_value * 1.2,
    )


    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    ax.set_zlabel(
        "z [m]"
    )


    ax.set_title(
        "Furuta Rotary Inverted Pendulum"
    )


    ax.set_box_aspect(
        [
            1.0,
            1.0,
            0.8,
        ]
    )


    # --------------------------------------------------------
    # Ground / pivot
    # --------------------------------------------------------

    pivot, = ax.plot(
        [0.0],
        [0.0],
        [0.0],
        marker="o",
        markersize=8,
    )


    # --------------------------------------------------------
    # Arm and pendulum
    # --------------------------------------------------------

    arm_line, = ax.plot(
        [],
        [],
        [],
        linewidth=4,
        label="rotary arm",
    )


    pendulum_line, = ax.plot(
        [],
        [],
        [],
        linewidth=4,
        label="pendulum",
    )


    pendulum_mass, = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=10,
    )


    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace_line, = ax.plot(
        [],
        [],
        [],
        linewidth=1,
        alpha=0.5,
    )


    trace_x = []
    trace_y = []
    trace_z = []


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    info_text = ax.text2D(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        family="monospace",
    )


    ax.legend(
        loc="upper right"
    )


    # ========================================================
    # GEOMETRY FUNCTION
    # ========================================================

    def furuta_geometry(
        theta_value,
        alpha_value,
    ):

        # ----------------------------------------------------
        # Arm endpoint
        # ----------------------------------------------------

        arm_x = (
            Lr_value
            * np.cos(
                theta_value
            )
        )

        arm_y = (
            Lr_value
            * np.sin(
                theta_value
            )
        )

        arm_z = 0.0


        # ----------------------------------------------------
        # Pendulum COM/end position
        #
        # Same convention used in symbolic model:
        #
        # alpha = 0   -> downward
        # alpha = pi  -> upright
        # ----------------------------------------------------

        pend_x = (
            arm_x

            - Lp_value
            * np.sin(
                alpha_value
            )
            * np.sin(
                theta_value
            )
        )


        pend_y = (
            arm_y

            + Lp_value
            * np.sin(
                alpha_value
            )
            * np.cos(
                theta_value
            )
        )


        pend_z = (
            -Lp_value
            * np.cos(
                alpha_value
            )
        )


        return (
            np.array([
                0.0,
                arm_x,
            ]),

            np.array([
                0.0,
                arm_y,
            ]),

            np.array([
                0.0,
                arm_z,
            ]),

            np.array([
                arm_x,
                pend_x,
            ]),

            np.array([
                arm_y,
                pend_y,
            ]),

            np.array([
                arm_z,
                pend_z,
            ]),

            pend_x,
            pend_y,
            pend_z,
        )


    # ========================================================
    # UPDATE FUNCTION
    # ========================================================

    def update_animation(
        frame_number,
    ):

        sample_index = (
            frame_indices[
                frame_number
            ]
        )


        theta_value = (
            x_history[
                sample_index,
                0
            ]
        )


        alpha_value = (
            x_history[
                sample_index,
                1
            ]
        )


        (
            arm_x,
            arm_y,
            arm_z,

            pend_x,
            pend_y,
            pend_z,

            tip_x,
            tip_y,
            tip_z,

        ) = furuta_geometry(
            theta_value,
            alpha_value,
        )


        # ----------------------------------------------------
        # Arm
        # ----------------------------------------------------

        arm_line.set_data(
            arm_x,
            arm_y,
        )

        arm_line.set_3d_properties(
            arm_z
        )


        # ----------------------------------------------------
        # Pendulum
        # ----------------------------------------------------

        pendulum_line.set_data(
            pend_x,
            pend_y,
        )

        pendulum_line.set_3d_properties(
            pend_z
        )


        # ----------------------------------------------------
        # Pendulum mass
        # ----------------------------------------------------

        pendulum_mass.set_data(
            [
                tip_x
            ],
            [
                tip_y
            ],
        )

        pendulum_mass.set_3d_properties(
            [
                tip_z
            ]
        )


        # ----------------------------------------------------
        # Trace
        # ----------------------------------------------------

        trace_x.append(
            tip_x
        )

        trace_y.append(
            tip_y
        )

        trace_z.append(
            tip_z
        )


        MAX_TRACE_POINTS = 120


        if (
            len(
                trace_x
            )
            > MAX_TRACE_POINTS
        ):

            del trace_x[
                0
            ]

            del trace_y[
                0
            ]

            del trace_z[
                0
            ]


        trace_line.set_data(
            trace_x,
            trace_y,
        )

        trace_line.set_3d_properties(
            trace_z
        )


        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        time_value = (
            times[
                sample_index
            ]
        )


        alpha_error_value = (
            np.rad2deg(
                wrap_to_pi(
                    alpha_value
                    - np.pi
                )
            )
        )


        theta_deg_value = (
            np.rad2deg(
                theta_value
            )
        )


        current_value = (
            x_history[
                sample_index,
                4
            ]
        )


        if (
            sample_index
            < N_SAMPLE
        ):

            duty_value = (
                duty_history[
                    sample_index
                ]
            )


            torque_value = (
                average_torque_history[
                    sample_index
                ]
            )

        else:

            duty_value = (
                duty_history[
                    -1
                ]
            )


            torque_value = (
                average_torque_history[
                    -1
                ]
            )


        info_text.set_text(

            f"t       = {time_value:6.3f} s\n"

            f"theta   = {theta_deg_value:+8.2f} deg\n"

            f"alpha-e = {alpha_error_value:+8.4f} deg\n"

            f"duty    = {duty_value:+8.4f}\n"

            f"current = {current_value:+8.4f} A\n"

            f"tau_avg = {torque_value:+8.4f} Nm"
        )


        return (
            arm_line,
            pendulum_line,
            pendulum_mass,
            trace_line,
            info_text,
        )


    # ========================================================
    # CREATE ANIMATION
    # ========================================================

    animation = FuncAnimation(

        fig,

        update_animation,

        frames=N_VIDEO_FRAMES,

        interval=(
            1000.0
            / VIDEO_FPS
        ),

        blit=False,
    )


    # ========================================================
    # SAVE MP4
    # ========================================================

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        print(
            "WARNING: ffmpeg was not found in PATH. "
            "Video was not generated."
        )
    else:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path

        writer = FFMpegWriter(
            fps=VIDEO_FPS,
            bitrate=2500,
        )

        animation.save(
            VIDEO_FILENAME,
            writer=writer,
            dpi=150,
        )

        print(
            f"Generated: {VIDEO_FILENAME}"
        )

    plt.close(fig)