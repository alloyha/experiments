r"""
Controlabilidade e Observabilidade em Sistemas Lineares MIMO
==============================================================

Para o sistema CONTÍNUO   dx/dt = A x + B u ,  y = C x
e o sistema DISCRETO      x[k+1] = A x[k] + B u[k] , y[k] = C x[k]

este script:

1) Verifica controlabilidade e observabilidade (rank das matrizes de
   Kalman).
2) Calcula o Gramiano de controlabilidade W_c e usa
        u(t) = B^T e^{A^T (t1-t)} W_c^{-1} [x1 - e^{A(t1-t0)} x0]
   (contínuo) ou
        u[k] = B^T (A^T)^{N-1-k} W_d^{-1} [x1 - A^N x0]
   (discreto)
   para levar o estado de x0 a x1 em tempo finito.
3) Calcula o Gramiano de observabilidade W_o e reconstrói o estado
   inicial a partir da saída medida y(t) (ou y[k]):
        x0_hat = W_o^{-1} \int C^T e^{A^T tau}... y  dtau  (contínuo)
        x0_hat = M_o^{-1} sum (A^T)^k C^T y[k]            (discreto)
   e então propaga x1_hat = e^{A(t1-t0)} x0_hat  (ou A^N x0_hat).
4) Plota tudo: trajetórias de estado, sinal de controle, saída medida
   e a comparação entre o estado real e o estimado.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp, quad_vec

np.set_printoptions(precision=4, suppress=True)

# ----------------------------------------------------------------------
# 1) DEFINIÇÃO DOS SISTEMAS MIMO (2 estados, 2 entradas, 2 saídas)
# ----------------------------------------------------------------------

# --- Sistema contínuo ---
Ac = np.array([[0.0, 1.0],
               [-2.0, -1.0]])
Bc = np.array([[1.0, 0.0],
               [0.0, 1.0]])
Cc = np.array([[1.0, 0.0],
               [0.0, 1.0]])

# --- Sistema discreto (obtido por discretização exata de Ac,Bc, T=0.2) ---
Ts = 0.2
Ad = expm(Ac * Ts)
Bd = np.linalg.solve(Ac, (Ad - np.eye(2)) @ Bc)  # B_d = A^{-1}(A_d - I) B
Cd = Cc.copy()

n = Ac.shape[0]

# ----------------------------------------------------------------------
# 2) TESTES DE CONTROLABILIDADE / OBSERVABILIDADE (posto de Kalman)
# ----------------------------------------------------------------------

def kalman_ctrb(A, B):
    n = A.shape[0]
    M = B.copy()
    Ak = np.eye(n)
    cols = [B]
    for k in range(1, n):
        Ak = Ak @ A
        cols.append(Ak @ B)
    return np.hstack(cols)

def kalman_obsv(A, C):
    return kalman_ctrb(A.T, C.T).T

for name, A, B, C in [("Contínuo", Ac, Bc, Cc), ("Discreto", Ad, Bd, Cd)]:
    Mc = kalman_ctrb(A, B)
    Mo = kalman_obsv(A, C)
    print(f"[{name}] rank controlabilidade = {np.linalg.matrix_rank(Mc)} / {n}"
          f"   rank observabilidade = {np.linalg.matrix_rank(Mo)} / {n}")

# ----------------------------------------------------------------------
# 3) CASO CONTÍNUO: Gramianos, u(t) e reconstrução x0_hat
# ----------------------------------------------------------------------

t0, t1 = 0.0, 3.0
x0 = np.array([2.0, -1.0])       # estado inicial real
x1_target = np.array([-1.0, 1.5])  # estado final desejado

def Wc_continuous(A, B, t0, t1):
    def integrand(tau):
        M = expm(A * (t1 - tau))
        return M @ B @ B.T @ M.T
    val, _ = quad_vec(integrand, t0, t1)
    return val

Wc = Wc_continuous(Ac, Bc, t0, t1)
Wc_inv = np.linalg.inv(Wc)

def u_continuous(t):
    # entrada que leva x0 -> x1_target no intervalo [t0, t1]
    vec = x1_target - expm(Ac * (t1 - t0)) @ x0
    return Bc.T @ expm(Ac.T * (t1 - t)) @ Wc_inv @ vec

def ode_rhs(t, x):
    return Ac @ x + Bc @ u_continuous(t)

t_eval = np.linspace(t0, t1, 400)
sol = solve_ivp(ode_rhs, (t0, t1), x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
x_traj_c = sol.y                      # (n, len(t_eval))
u_traj_c = np.array([u_continuous(t) for t in t_eval]).T

print(f"\n[Contínuo] x(t1) obtido   = {x_traj_c[:, -1]}")
print(f"[Contínuo] x1 desejado    = {x1_target}")

# --- Observabilidade contínua: reconstruir x0 a partir de y(t) ---
# Resposta livre (u=0) a partir do MESMO x0, para servir de "medição"
def ode_free(t, x):
    return Ac @ x
sol_free = solve_ivp(ode_free, (t0, t1), x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
x_free_c = sol_free.y
y_meas_c = Cc @ x_free_c              # saída medida y(t) = C x(t)

def Wo_continuous(A, C, t0, t1):
    def integrand(tau):
        M = expm(A.T * (tau - t0))
        return M @ C.T @ C @ M.T
    val, _ = quad_vec(integrand, t0, t1)
    return val

Wo = Wo_continuous(Ac, Cc, t0, t1)
Wo_inv = np.linalg.inv(Wo)

def integrand_obs(tau_idx):
    tau = t_eval[tau_idx]
    M = expm(Ac.T * (tau - t0))
    return M @ Cc.T @ y_meas_c[:, tau_idx]

integrand_vals = np.array([integrand_obs(i) for i in range(len(t_eval))]).T
x0_hat_c = Wo_inv @ np.trapezoid(integrand_vals, t_eval, axis=1)
x1_hat_c = expm(Ac * (t1 - t0)) @ x0_hat_c

print(f"[Contínuo] x0 real        = {x0}")
print(f"[Contínuo] x0 estimado    = {x0_hat_c}")
print(f"[Contínuo] x1 estimado    = {x1_hat_c}")

# ----------------------------------------------------------------------
# 4) CASO DISCRETO: Gramianos, u[k] e reconstrução x0_hat
# ----------------------------------------------------------------------

N = n  # número mínimo de passos (deadbeat), pode ser aumentado
x0_d = np.array([2.0, -1.0])
x1_target_d = np.array([-1.0, 1.5])

def Wd_discrete(A, B, N):
    W = np.zeros((A.shape[0], A.shape[0]))
    Ak = np.eye(A.shape[0])
    for k in range(N):
        W += Ak @ B @ B.T @ Ak.T
        Ak = Ak @ A
    return W

Wd = Wd_discrete(Ad, Bd, N)
Wd_inv = np.linalg.inv(Wd)

def u_discrete_seq(A, B, x0, x1, N):
    Wd_ = Wd_discrete(A, B, N)
    Wd_inv_ = np.linalg.inv(Wd_)
    AN = np.linalg.matrix_power(A, N)
    vec = x1 - AN @ x0
    us = []
    for k in range(N):
        Ak_pow = np.linalg.matrix_power(A.T, N - 1 - k)
        us.append(B.T @ Ak_pow @ Wd_inv_ @ vec)
    return np.array(us).T  # shape (m, N)

u_seq_d = u_discrete_seq(Ad, Bd, x0_d, x1_target_d, N)

x_traj_d = np.zeros((n, N + 1))
x_traj_d[:, 0] = x0_d
for k in range(N):
    x_traj_d[:, k + 1] = Ad @ x_traj_d[:, k] + Bd @ u_seq_d[:, k]

print(f"\n[Discreto] x[N] obtido    = {x_traj_d[:, -1]}")
print(f"[Discreto] x1 desejado    = {x1_target_d}")

# --- Observabilidade discreta: reconstruir x[0] a partir de y[k] ---
Nobs = n
x_free_d = np.zeros((n, Nobs))
x_free_d[:, 0] = x0_d
for k in range(1, Nobs):
    x_free_d[:, k] = Ad @ x_free_d[:, k - 1]
y_meas_d = Cd @ x_free_d

def Mo_discrete(A, C, N):
    M = np.zeros((A.shape[0], A.shape[0]))
    Ak = np.eye(A.shape[0])
    for k in range(N):
        M += Ak.T @ C.T @ C @ Ak
        Ak = Ak @ A
    return M

Mo = Mo_discrete(Ad, Cd, Nobs)
Mo_inv = np.linalg.inv(Mo)

acc = np.zeros(n)
Ak = np.eye(n)
for k in range(Nobs):
    acc += Ak.T @ Cd.T @ y_meas_d[:, k]
    Ak = Ak @ Ad
x0_hat_d = Mo_inv @ acc
x1_hat_d = np.linalg.matrix_power(Ad, Nobs) @ x0_hat_d

print(f"[Discreto] x0 real        = {x0_d}")
print(f"[Discreto] x0 estimado    = {x0_hat_d}")
print(f"[Discreto] x1 estimado    = {x1_hat_d}")

# ----------------------------------------------------------------------
# 5) GRÁFICOS
# ----------------------------------------------------------------------

fig, axs = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Controlabilidade e Observabilidade — Sistemas MIMO Contínuo x Discreto",
             fontsize=13, fontweight="bold")

# --- (0,0) Estados contínuos ---
ax = axs[0, 0]
ax.plot(t_eval, x_traj_c[0], label=r"$x_1(t)$")
ax.plot(t_eval, x_traj_c[1], label=r"$x_2(t)$")
ax.axhline(x1_target[0], ls="--", color="C0", alpha=0.5)
ax.axhline(x1_target[1], ls="--", color="C1", alpha=0.5)
ax.scatter([t0, t0], x0, color="k", zorder=5, label="x(t0)")
ax.scatter([t1, t1], x1_target, color="r", marker="x", zorder=5, label="x1 alvo")
ax.set_title("Contínuo: estado sob controle u(t)\n(leva x0 → x1)")
ax.set_xlabel("t [s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (0,1) Entrada contínua ---
ax = axs[0, 1]
ax.plot(t_eval, u_traj_c[0], label=r"$u_1(t)$")
ax.plot(t_eval, u_traj_c[1], label=r"$u_2(t)$")
ax.set_title("Contínuo: sinal de controle u(t)")
ax.set_xlabel("t [s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (0,2) Observabilidade contínua ---
ax = axs[0, 2]
ax.plot(t_eval, y_meas_c[0], label=r"$y_1(t)$ medido")
ax.plot(t_eval, y_meas_c[1], label=r"$y_2(t)$ medido")
ax.scatter([t0, t0], x0_hat_c, color="g", marker="^", zorder=5, label=r"$\hat{x}(t_0)$")
ax.scatter([t0, t0], x0, color="k", marker="o", zorder=5, s=20, label="x(t0) real")
ax.set_title("Contínuo: saída medida e\nestado reconstruído")
ax.set_xlabel("t [s]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (1,0) Estados discretos ---
ax = axs[1, 0]
k_axis = np.arange(N + 1)
ax.step(k_axis, x_traj_d[0], where="post", marker="o", label=r"$x_1[k]$")
ax.step(k_axis, x_traj_d[1], where="post", marker="o", label=r"$x_2[k]$")
ax.axhline(x1_target_d[0], ls="--", color="C0", alpha=0.5)
ax.axhline(x1_target_d[1], ls="--", color="C1", alpha=0.5)
ax.scatter([0, 0], x0_d, color="k", zorder=5)
ax.scatter([N, N], x1_target_d, color="r", marker="x", zorder=5)
ax.set_title(f"Discreto: estado sob controle u[k]\n(leva x0 → x1 em N={N} passos)")
ax.set_xlabel("k"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (1,1) Entrada discreta ---
ax = axs[1, 1]
ax.step(np.arange(N), u_seq_d[0], where="post", marker="o", label=r"$u_1[k]$")
ax.step(np.arange(N), u_seq_d[1], where="post", marker="o", label=r"$u_2[k]$")
ax.set_title("Discreto: sinal de controle u[k]")
ax.set_xlabel("k"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- (1,2) Observabilidade discreta ---
ax = axs[1, 2]
ax.step(np.arange(Nobs), y_meas_d[0], where="post", marker="s", label=r"$y_1[k]$ medido")
ax.step(np.arange(Nobs), y_meas_d[1], where="post", marker="s", label=r"$y_2[k]$ medido")
ax.scatter([0, 0], x0_hat_d, color="g", marker="^", zorder=5, label=r"$\hat{x}[0]$")
ax.scatter([0, 0], x0_d, color="k", marker="o", zorder=5, s=20, label="x[0] real")
ax.set_title("Discreto: saída medida e\nestado reconstruído")
ax.set_xlabel("k"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("/home/claude/mimo_ctrl_obs.png", dpi=150)
print("\nGráfico salvo em mimo_ctrl_obs.png")

