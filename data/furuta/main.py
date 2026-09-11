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
       controlador transverso discreto.

8. Estimação:
       Kalman discreto usando somente theta.

Objetivo:
       alpha -> pi
       alpha_dot -> 0

A zero manifold não necessariamente impõe:
       theta -> 0
       theta_dot -> 0
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

# Tempo de amostragem e frequência de amostragem
T_SAMPLE = 0.002
F_SAMPLE = 1.0 / T_SAMPLE

# Tempo total de simulação
SIMULATION_TIME = 15.0

# Duty cycle máximo permitido
MAX_DUTY = 1.0


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

GEAR_RATIO = 1.0
GEAR_EFFICIENCY = 1.0


TAU_ELECTRICAL = (
    L_MOTOR / R_MOTOR
)


# ============================================================
# 4. CONTROLADOR UNIFICADO TANGENCIAL-TRANSVERSAL (LQR)
#
# Em vez de dois controladores desconectados
# (feedforward V_u*eta  +  K_perp*z_perp via pole placement),
# sintetizamos um UNICO ganho
#
#       K_xi = [ K_eta   K_perp ]
#
# em coordenadas xi = [eta ; z_perp], via LQR discreto, com
#
#       Q_xi = blockdiag(Q_ETA_WEIGHT, Q_ZPERP_WEIGHT)
#
# Q_ZPERP_WEIGHT >> Q_ETA_WEIGHT reflete a prioridade:
# primeiro nao deixar o pendulo cair (z_perp -> 0 rapido),
# depois, mais devagar, trazer o braco a zero (eta -> 0).
#
# Como ha um unico atuador, essa regulacao tangencial deixa
# de preservar a zero manifold de forma EXATA (z_perp=0 nao
# e mais exatamente invariante quando K_eta != 0) — essa e a
# troca conceitual discutida: exatidao da manifold versus
# convergencia completa ao equilibrio x_eq_full.
# ============================================================

Q_ETA_WEIGHT = np.diag([
    1.0,      # theta    (regulacao lenta, baixa prioridade)
    1.0,      # theta_dot
])

Q_ZPERP_WEIGHT = np.diag([
    1.0e4,    # alpha       (alta prioridade: nao deixar cair)
    1.0e2,    # alpha_dot
    1.0e2,    # current
])

R_XI = np.array([
    [1.0]
])


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

params = {

    m: 0.127,

    Lr: 0.2159,
    Lp: 0.1683,

    Jr: 0.0050,
    Jp: 0.0012,

    br: 0.002,
    bp: 0.001,

    g: 9.81,
}


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
# 23. CONTROLADOR UNIFICADO TANGENCIAL-TRANSVERSAL (LQR)
#
# Transformacao de coordenadas:
#
#       x = T_XI @ xi,     xi = [eta ; z_perp],
#       T_XI = [ V_x   Q_perp ]   (5x5)
#
# Note que T_XI NAO precisa ser ortonormal para ser valida —
# apenas invertivel. V_x da a base (nao-ortonormal) do
# subespaco tangente; Q_perp ja e ortonormal (via null_space).
# ============================================================

T_XI = np.hstack([
    V_x,
    Q_perp,
])


T_XI_COND = np.linalg.cond(
    T_XI
)


T_XI_INV = np.linalg.inv(
    T_XI
)


A_xi_c = (
    T_XI_INV
    @ A_full
    @ T_XI
)


B_xi_c = (
    T_XI_INV
    @ B_full
)


N_XI = (
    A_xi_c.shape[0]
)


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

        np.eye(
            N_XI
        ),

        np.zeros(
            (
                N_XI,
                1,
            )
        ),
    ),

    T_SAMPLE,

    method="zoh",
)


# --------------------------------------------------------
# LQR discreto conjunto
# --------------------------------------------------------

Q_XI = (
    block_diag(
        Q_ETA_WEIGHT,
        Q_ZPERP_WEIGHT,
    )
)


P_XI = solve_discrete_are(
    A_xi_d,
    B_xi_d,
    Q_XI,
    R_XI,
)


K_XI = (
    np.linalg.inv(
        R_XI
        + B_xi_d.T
        @ P_XI
        @ B_xi_d
    )

    @ (
        B_xi_d.T
        @ P_XI
        @ A_xi_d
    )
)


K_eta = (
    K_XI[:, :N_ZERO]
)


K_perp = (
    K_XI[:, N_ZERO:]
)


# --------------------------------------------------------
# Polos de malha fechada resultantes (para diagnostico) —
# equivalente discreto de "TRANSVERSE_CONTINUOUS_POLES" do
# projeto anterior, mas agora sao consequencia do LQR, nao
# uma escolha direta.
# --------------------------------------------------------

closed_loop_poles_xi, _ = (
    eig(
        A_xi_d
        - B_xi_d
        @ K_XI,
    )
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
# 27. CONTROLADOR
# ============================================================

def control_law(
    dx_hat_red,
    previous_duty,
):

    dx_hat_aug = (
        augmented_estimate(
            dx_hat_red,
            previous_duty,
        )
    )


    eta_hat = (
        V_x_pinv
        @ dx_hat_aug
    )


    z_hat = (
        Q_perp.T
        @ dx_hat_aug
    )


    # Controlador unificado K_xi = [K_eta  K_perp] (LQR),
    # sintetizado em coordenadas xi = [eta ; z_perp].
    # Substitui o antigo par (feedforward V_u@eta) + (K_perp
    # via pole placement) por um unico ganho projetado
    # conjuntamente, com Q_ZPERP_WEIGHT >> Q_ETA_WEIGHT.

    duty_eta = float(
        (
            -K_eta
            @ eta_hat
        ).item()
    )


    duty_transverse = float(
        (
            -K_perp
            @ z_hat
        ).item()
    )


    duty_raw = (
        duty_eta
        + duty_transverse
    )


    duty = (
        saturate_duty(
            duty_raw
        )
    )


    return (
        duty,
        duty_raw,
        duty_eta,
        duty_transverse,
        eta_hat,
        z_hat,
        dx_hat_aug,
    )


# ============================================================
# 28. KALMAN PREDICT / CORRECT
# ============================================================

def kalman_step(
    dx_hat,
    duty,
    measurement_next,
):

    prediction = (
        A_red_d
        @ dx_hat

        + B_red_d[:, 0]
        * duty
    )


    innovation = (
        measurement_next

        - float(
            (
                C_measure
                @ prediction
            ).item()
        )
    )


    corrected = (
        prediction

        + K_kalman[:, 0]
        * innovation
    )


    return (
        corrected,
        innovation,
    )


# ============================================================
# 29. CONDIÇÕES INICIAIS
# ============================================================

x_initial = np.array([

    np.deg2rad(
        33.0
    ),

    np.pi
    + np.deg2rad(
        8.0
    ),

    np.deg2rad(
        7.0
    ),

    np.deg2rad(
        1.0
    ),

    0.0,
])


dx_hat_initial = np.array([
    x_initial[0],
    0.0,
    0.0,
    0.0,
])


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


duty_raw_history = np.zeros(
    N_SAMPLE
)


duty_eta_history = np.zeros(
    N_SAMPLE
)


duty_transverse_history = np.zeros(
    N_SAMPLE
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


print("\nQ_XI (pesos do LQR, blockdiag(Q_eta, Q_zperp)) =")
print(
    Q_XI
)


print("\nK_XI = [K_eta  K_perp] (ganho unico, LQR) =")
print(
    K_XI
)


print("\nK_eta =")
print(
    K_eta
)


print("\nK_perp =")
print(
    K_perp
)


print("\nPolos de malha fechada em xi (discretos, |z|<1 para estabilidade):")
print(
    closed_loop_poles_xi
)

print(
    "  |polos| =",
    np.abs(
        closed_loop_poles_xi
    ),
)


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

    (
        duty,
        duty_raw,
        duty_eta,
        duty_transverse,
        _,
        _,
        dx_hat_aug,

    ) = control_law(
        dx_hat_k,
        previous_duty,
    )


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


    dx_hat_aug_history[k] = (
        dx_hat_aug
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
    dx_true_history[:, :4]
    - dx_hat_red_history
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

]:

    print(
        " ",
        filename,
    )

# ============================================================
# VIDEO ANIMATION
# ============================================================

from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


SAVE_VIDEO = True

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

    writer = FFMpegWriter(

        fps=VIDEO_FPS,

        bitrate=2500,
    )


    animation.save(
        VIDEO_FILENAME,
        writer=writer,
        dpi=150,
    )


    plt.close(
        fig
    )


    print(
        f"Generated: {VIDEO_FILENAME}"
    )
