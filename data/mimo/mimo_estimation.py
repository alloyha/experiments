r"""
Controle por realimentação de estado ESTIMADO, com observador de
Luenberger rodando SIMULTANEAMENTE (não mais em duas fases separadas).

  Planta:      x[k+1] = A x[k] + B u[k],      y[k] = C x[k]
  Observador:  xhat[k+1] = A xhat[k] + B u[k] + L (y[k] - C xhat[k])
  Controle:    u[k] = u_ss - K (xhat[k] - x_target)

onde x_target é um ponto de equilíbrio da malha fechada:
      x_target = A x_target + B u_ss   =>   u_ss = B^{-1}(I-A) x_target

PRINCÍPIO DA SEPARAÇÃO
-----------------------
Definindo o erro de rastreamento  xi[k] = x[k] - x_target
e o erro de estimação             e[k]  = x[k] - xhat[k],
a dinâmica conjunta é bloco-triangular:

    [xi[k+1]]   [A-BK   BK ] [xi[k]]
    [e[k+1] ] = [ 0    A-LC] [e[k] ]

Os autovalores da malha fechada são a UNIÃO de eig(A-BK) e eig(A-LC):
K e L podem ser projetados de forma INDEPENDENTE. Aqui usamos alocação
"deadbeat" (autovalores em 0) para os dois, o que garante convergência
exata em no máximo 2n passos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.signal import place_poles
from scipy.integrate import solve_ivp

np.set_printoptions(precision=4, suppress=True)

# ----------------------------------------------------------------------
# 1) Sistema (mesmo de ordem 2, MIMO 2x2)
# ----------------------------------------------------------------------
Ac = np.array([[0.0, 1.0],
               [-2.0, -1.0]])
Bc = np.array([[1.0, 0.0],
               [0.0, 1.0]])
Cc = np.array([[1.0, 0.0]])          # <-- saída diferente: mede-se só a posição x1
n, m, p = 2, 2, 1

# didático: confere que (A,C) ainda é observável mesmo sem medir x2 diretamente
Mo_check = np.vstack([Cc, Cc @ Ac])
print(f"Matriz de observabilidade [C; CA] =\n{Mo_check}")
print(f"rank = {np.linalg.matrix_rank(Mo_check)} / {n}  "
      f"({'observável' if np.linalg.matrix_rank(Mo_check)==n else 'NÃO observável'})\n")

Ts = 0.3
Ad = expm(Ac * Ts)
Bd = np.linalg.solve(Ac, (Ad - np.eye(n)) @ Bc)
Cd = Cc.copy()

x0_real = np.array([2.0, -1.0])
x_target = np.array([-1.0, 1.5])
xhat0 = np.array([0.0, 0.0])          # estimativa inicial (sem informação prévia)

# ----------------------------------------------------------------------
# 2) Projeto INDEPENDENTE de K (controlador) e L (observador) -- deadbeat
# ----------------------------------------------------------------------
poles_ctrl = [0.0, 0.02]     # ligeiramente distintos p/ robustez numérica
poles_obs = [0.0, -0.02]

K = place_poles(Ad, Bd, poles_ctrl).gain_matrix
L = place_poles(Ad.T, Cd.T, poles_obs).gain_matrix.T

print("Autovalores de A-BK (dinâmica do controlador):",
      np.round(np.linalg.eigvals(Ad - Bd @ K), 4))
print("Autovalores de A-LC (dinâmica do observador):  ",
      np.round(np.linalg.eigvals(Ad - L @ Cd), 4))

# ponto de equilíbrio: x_target = A x_target + B u_ss
u_ss = np.linalg.solve(Bd, (np.eye(n) - Ad) @ x_target)
print(f"\nu_ss (entrada de equilíbrio p/ manter x_target) = {u_ss}\n")

# ----------------------------------------------------------------------
# 3) Simulação simultânea: observador + controle rodando juntos
# ----------------------------------------------------------------------
N = 12   # número de passos simulados

x = np.zeros((n, N + 1))
xhat = np.zeros((n, N + 1))
u_hist = np.zeros((m, N))
x[:, 0] = x0_real
xhat[:, 0] = xhat0

for k in range(N):
    uk = u_ss - K @ (xhat[:, k] - x_target)
    yk = Cd @ x[:, k]

    x[:, k + 1] = Ad @ x[:, k] + Bd @ uk
    xhat[:, k + 1] = Ad @ xhat[:, k] + Bd @ uk + L @ (yk - Cd @ xhat[:, k])

    u_hist[:, k] = uk

xi_norm = np.linalg.norm(x - x_target[:, None], axis=0)
e_norm = np.linalg.norm(x - xhat, axis=0)

print("k | ||x-x_target||   ||x-xhat||")
for k in range(N + 1):
    print(f"{k:2d} | {xi_norm[k]:14.6f}   {e_norm[k]:.6f}")

# ----------------------------------------------------------------------
# 4) Sinal contínuo x(t), com u(t) em ZOH, sobreposto ao discreto
# ----------------------------------------------------------------------
t_cont, x_cont = [], []
x0_int = x0_real.copy()
for k in range(N):
    tk0, tk1 = k * Ts, (k + 1) * Ts
    uk = u_hist[:, k]

    def rhs(t, x, uk=uk):
        return Ac @ x + Bc @ uk

    t_eval = np.linspace(tk0, tk1, 25)
    sol = solve_ivp(rhs, (tk0, tk1), x0_int, t_eval=t_eval, rtol=1e-10, atol=1e-10)
    t_cont.append(sol.t)
    x_cont.append(sol.y)
    x0_int = sol.y[:, -1]

t_cont = np.concatenate(t_cont)
x_cont = np.concatenate(x_cont, axis=1)
t_disc = np.arange(N + 1) * Ts

# ----------------------------------------------------------------------
# 5) Gráficos
# ----------------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
fig.suptitle("Controle com estado estimado simultâneo — convergência via "
             "princípio da separação", fontsize=12, fontweight="bold")

# --- estado: contínuo sobreposto ao discreto, real e estimado ---
ax = axs[0]
ax.plot(t_cont, x_cont[0], color="C0", lw=1.5, label=r"$x_1(t)=y(t)$ contínuo — MEDIDO")
ax.plot(t_disc, x[0], "o", color="C0", ms=6, mfc="white", mew=1.8, label=r"$x_1[k]=y[k]$ medido")
ax.plot(t_disc, xhat[0], "^", color="C0", ms=6, alpha=0.6, label=r"$\hat x_1[k]$ estimado")
ax.plot(t_cont, x_cont[1], color="C1", lw=1.5, label=r"$x_2(t)$ contínuo — NÃO medido")
ax.plot(t_disc, x[1], "s", color="C1", ms=6, mfc="white", mew=1.8, label=r"$x_2[k]$ real (oculto)")
ax.plot(t_disc, xhat[1], "^", color="C1", ms=6, alpha=0.6, label=r"$\hat x_2[k]$ reconstruído")
ax.axhline(x_target[0], color="C0", ls="--", alpha=0.3)
ax.axhline(x_target[1], color="C1", ls="--", alpha=0.3)
ax.set_xlabel("t [s]")
ax.set_ylabel("estado")
ax.set_title(r"$x_1$ é medido diretamente (sensor); $x_2$ é reconstruído pelo observador")
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# --- erros em escala log: separação em ação ---
ax = axs[1]
k_axis = np.arange(N + 1)
ax.semilogy(k_axis, xi_norm + 1e-16, "o-", color="C3",
            label=r"$\|x[k]-x_{target}\|$ (erro de rastreamento, eig(A-BK))")
ax.semilogy(k_axis, e_norm + 1e-16, "s-", color="C4",
            label=r"$\|x[k]-\hat x[k]\|$ (erro de estimação, eig(A-LC))")
ax.set_xlabel("k")
ax.set_ylabel("norma do erro (log)")
ax.set_title("Convergência simultânea e independente (princípio da separação)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")

# --- controle ---
ax = axs[2]
ax.step(np.arange(N), u_hist[0], where="post", marker="o", color="C2", label=r"$u_1[k]$")
ax.step(np.arange(N), u_hist[1], where="post", marker="o", color="C5", label=r"$u_2[k]$")
ax.axhline(u_ss[0], color="C2", ls=":", alpha=0.5, label=r"$u_{1,ss}$")
ax.axhline(u_ss[1], color="C5", ls=":", alpha=0.5, label=r"$u_{2,ss}$")
ax.set_xlabel("k")
ax.set_ylabel("entrada")
ax.set_title(r"Controle $u[k] = u_{ss} - K(\hat x[k]-x_{target})$")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("mimo_separacao.png", dpi=150)
print("\nGráfico salvo.")

