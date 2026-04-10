/*
 * test.c — Automated physics test suite
 *
 * Build:  make test_physics
 * Run:    ./test_physics
 *
 * Each test prints PASS / FAIL with a description.
 * Exit code is the number of failures (0 = all passed).
 */

#include "physics.h"
#include "bodies.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════════════════════════════════════════════════════
 * Minimal test framework
 * ═══════════════════════════════════════════════════════════════ */

static int _pass = 0, _fail = 0;

#define GREEN  "\033[32m"
#define RED    "\033[31m"
#define YELLOW "\033[33m"
#define RESET  "\033[0m"

static void test_section(const char *name)
{
    printf("\n%s── %s %s\n", YELLOW, name, RESET);
}

#define ASSERT_NEAR(actual, expected, tol, desc)                        \
    do {                                                                \
        double _a = (actual), _e = (expected), _t = (tol);             \
        double _err = fabs(_a - _e);                                    \
        double _rel = (_e != 0.0) ? _err / fabs(_e) : _err;           \
        if (_rel <= _t || _err <= _t) {                                 \
            printf(GREEN "  PASS" RESET "  %-52s  "                    \
                   "got %.6g  (expected %.6g)\n", desc, _a, _e);        \
            _pass++;                                                     \
        } else {                                                        \
            printf(RED   "  FAIL" RESET "  %-52s  "                    \
                   "got %.6g  (expected %.6g, err %.2e)\n",             \
                   desc, _a, _e, _rel);                                 \
            _fail++;                                                     \
        }                                                               \
    } while(0)

#define ASSERT_TRUE(cond, desc)                                         \
    do {                                                                \
        if (cond) {                                                     \
            printf(GREEN "  PASS" RESET "  %s\n", desc);               \
            _pass++;                                                    \
        } else {                                                        \
            printf(RED   "  FAIL" RESET "  %s\n", desc);               \
            _fail++;                                                    \
        }                                                               \
    } while(0)

/* ═══════════════════════════════════════════════════════════════
 * Test helpers
 * ═══════════════════════════════════════════════════════════════ */

/* Single-body scenario for isolated tests */
static Body make_earth(void)
{
    return (Body){
        .pos    = v2(0, 0),
        .mass   = EARTH_MASS,
        .radius = EARTH_RADIUS,
        .name   = "Earth",
        .symbol = 'E',
    };
}

static Body make_moon(void)
{
    return (Body){
        .pos    = v2(EARTH_MOON_DIST, 0),
        .mass   = MOON_MASS,
        .radius = MOON_RADIUS,
        .name   = "Moon",
        .symbol = 'L',
    };
}

/*
 * Specific orbital energy (vis-viva, per unit mass):
 *   ε = v² / 2 − G·M / r
 * For Newtonian gravity this is conserved along any trajectory.
 */
static double orbital_energy(const Spacecraft *sc,
                             const Body       *bodies,
                             int               n)
{
    double ke = 0.5 * v2_len(sc->vel) * v2_len(sc->vel);
    double pe = 0.0;
    for (int i = 0; i < n; i++) {
        double r = v2_dist(sc->pos, bodies[i].pos);
        if (r > EPSILON)
            pe -= G_CONST * bodies[i].mass / r;
    }
    return ke + pe;
}

/* ═══════════════════════════════════════════════════════════════
 * Test groups
 * ═══════════════════════════════════════════════════════════════ */

/* ── 1. Vector math ──────────────────────────────────────────── */
static void test_vec2(void)
{
    test_section("Vec2 arithmetic");

    Vec2 a = v2(3, 4);
    ASSERT_NEAR(v2_len(a), 5.0, 1e-12, "v2_len({3,4}) == 5");

    Vec2 b  = v2(1, 0);
    Vec2 nb = v2_norm(b);
    ASSERT_NEAR(v2_len(nb), 1.0, 1e-12, "norm of unit vector stays unit");

    Vec2 zero = v2(0, 0);
    Vec2 nz   = v2_norm(zero);
    ASSERT_NEAR(v2_len(nz), 0.0, 1e-12, "norm of zero vector is zero (no NaN)");

    Vec2 sum = v2_add(v2(1, 2), v2(3, 4));
    ASSERT_NEAR(sum.x, 4.0, 1e-12, "v2_add x");
    ASSERT_NEAR(sum.y, 6.0, 1e-12, "v2_add y");

    Vec2 sc = v2_scale(v2(2, 3), 2.5);
    ASSERT_NEAR(sc.x, 5.0, 1e-12, "v2_scale x");
    ASSERT_NEAR(sc.y, 7.5, 1e-12, "v2_scale y");

    ASSERT_NEAR(v2_dist(v2(0,0), v2(3,4)), 5.0, 1e-12, "v2_dist({0,0},{3,4}) == 5");
}

/* ── 2. Gravitational acceleration ───────────────────────────── */
static void test_grav_acc(void)
{
    test_section("Gravitational acceleration");

    Body earth = make_earth();

    /*
     * At Earth's surface (r = EARTH_RADIUS km), the acceleration
     * magnitude should match standard gravity:
     *   g = 9.80665 m/s²
     *     = 9.80665e-3 km/s²
     *     × 3600² s²/h²
     *     = 127,065 km/h²  (≈ 9.8 m/s² × 3600²/1000)
     *
     * We compute: G·M / R²  using our unit-adjusted G.
     */
    double g_expected = G_CONST * EARTH_MASS / (EARTH_RADIUS * EARTH_RADIUS);
    Vec2   pos_surface = v2(EARTH_RADIUS, 0);
    Vec2   acc         = grav_acc(&earth, pos_surface);

    /* Direction must point toward Earth centre (−x) */
    ASSERT_TRUE(acc.x < 0, "surface accel points toward Earth (−x)");
    ASSERT_NEAR(acc.y, 0.0, 1e-9, "surface accel has zero y component");
    ASSERT_NEAR(fabs(acc.x), g_expected, 1e-6,
                "surface |a| matches G·M/R²");

    /* Real-world sanity check: should be ~127 000 km/h² */
    ASSERT_NEAR(fabs(acc.x), 127065.0, 0.005,
                "|a_surface| ≈ 127 065 km/h²  (9.8 m/s² × 3600²/1000)");

    /* At 2× radius, acceleration should be 1/4 (inverse-square) */
    Vec2 acc2r = grav_acc(&earth, v2(2 * EARTH_RADIUS, 0));
    ASSERT_NEAR(fabs(acc2r.x) / fabs(acc.x), 0.25, 1e-6,
                "accel at 2R is 1/4 of accel at R (inverse-square)");

    /* Singularity guard: at body centre, return zero (no crash) */
    Vec2 acc_zero = grav_acc(&earth, earth.pos);
    ASSERT_NEAR(v2_len(acc_zero), 0.0, 1e-12,
                "grav_acc at body position returns zero (no NaN/inf)");
}

/* ── 3. Resultant acceleration (superposition) ───────────────── */
static void test_resultant_acc(void)
{
    test_section("Resultant acceleration (two-body superposition)");

    Body bodies[2] = { make_earth(), make_moon() };

    /* At Earth's centre, Moon's pull should dominate on the +x side */
    Vec2 a_moon_only = grav_acc(&bodies[1], bodies[0].pos);
    Vec2 a_total     = resultant_acc(bodies, 2, bodies[0].pos);

    /* Earth is at its own singularity → contributes 0, Moon contributes a_moon */
    ASSERT_NEAR(a_total.x, a_moon_only.x, 1e-9,
                "at Earth centre, total = Moon pull only");

    /*
     * Midpoint between Earth and Moon: both bodies pull in OPPOSITE
     * directions.  If m_E >> m_L the net force points toward Earth.
     */
    Vec2 mid    = v2(EARTH_MOON_DIST / 2.0, 0);
    Vec2 a_mid  = resultant_acc(bodies, 2, mid);
    ASSERT_TRUE(a_mid.x < 0,
                "at midpoint, net force points toward Earth (−x)");
}

/* ── 4. Escape velocity ──────────────────────────────────────── */
static void test_escape_velocity(void)
{
    test_section("Escape velocity");

    Body earth = make_earth();

    /*
     * Real Earth escape velocity at surface:
     *   11.186 km/s = 40 270 km/h
     * Our G is fitted to km/h, so we get the km/h value directly.
     */
    double v_esc = escape_velocity(&earth, v2(EARTH_RADIUS, 0));
    ASSERT_NEAR(v_esc, 40270.0, 0.005,    /* 0.5 % tolerance */
                "escape velocity at Earth surface ≈ 40 270 km/h (11.19 km/s)");

    /* v_esc² = 2·G·M/R — check the algebraic identity */
    double v2_theory = 2.0 * G_CONST * EARTH_MASS / EARTH_RADIUS;
    ASSERT_NEAR(v_esc * v_esc, v2_theory, 1e-9,
                "v_esc² == 2·G·M/R  (algebraic identity)");

    /* At 4× radius, v_esc should be halved (1/√r) */
    double v4r  = escape_velocity(&earth, v2(4 * EARTH_RADIUS, 0));
    ASSERT_NEAR(v4r / v_esc, 0.5, 1e-6,
                "v_esc at 4R is half v_esc at R  (1/√r law)");
}

/* ── 5. Lagrange L1 point ────────────────────────────────────── */
static void test_lagrange_l1(void)
{
    test_section("Lagrange L1 point");

    Body earth = make_earth();
    Body moon  = make_moon();

    Vec2 l1 = lagrange_l1(&earth, &moon);

    /*
     * Known value: L1 lies ~326 400 km from Earth centre
     * (Hill-sphere approximation: D·(1 − ∛(m_Moon/(3·m_Earth)))).
     * SOHO orbits near this point.
     */
    double d_earth = v2_dist(earth.pos, l1);
    double d_moon  = v2_dist(moon.pos, l1);

    /* L1 is between Earth and Moon */
    ASSERT_TRUE(l1.x > 0 && l1.x < EARTH_MOON_DIST,
                "L1 lies between Earth and Moon");
    ASSERT_NEAR(d_earth, 326000.0, 0.01,   /* 1 % tolerance */
                "L1 is ~326 000 km from Earth");

    /* L1 is closer to the Moon than to Earth */
    ASSERT_TRUE(d_moon < d_earth,
                "L1 is closer to Moon than to Earth");

    /*
     * The Hill-sphere L1 is an approximation valid in the CO-ROTATING frame.
     * In the inertial frame the forces do NOT cancel at this point —
     * the centrifugal pseudo-force is missing.  We instead verify that
     * the residual acceleration is much smaller than at the midpoint,
     * confirming the point is "near" the true saddle.
     */
    Body two_bodies[2] = { make_earth(), make_moon() };
    Vec2 a_mid   = resultant_acc(two_bodies, 2, v2(EARTH_MOON_DIST/2.0, 0));
    Vec2 a_at_l1 = resultant_acc(two_bodies, 2, l1);
    ASSERT_TRUE(v2_len(a_at_l1) < v2_len(a_mid),
                "net accel at L1 smaller than at midpoint (near saddle)");
}

/* ── 6. Velocity-Verlet integration: energy conservation ─────── */
static void test_verlet_energy(void)
{
    test_section("Velocity-Verlet — energy conservation");

    Body earth = make_earth();

    /*
     * Set up a circular orbit at r = 50 000 km above Earth centre.
     * Circular orbit speed: v_c = √(G·M/r)
     */
    double r  = 50000.0;
    double vc = sqrt(G_CONST * EARTH_MASS / r);

    Spacecraft sc;
    sc.pos               = v2(r, 0);
    sc.vel               = v2(0, vc);
    sc.acc               = grav_acc(&earth, sc.pos);
    sc.distance_traveled = 0.0;
    sc.time              = 0.0;

    double E0 = orbital_energy(&sc, &earth, 1);

    /* Integrate for 100 full orbital periods */
    double T  = 2.0 * M_PI * r / vc;        /* orbital period (h)  */
    double dt = T / 500.0;                   /* 500 steps per orbit */
    int    N  = 100 * 500;                   /* 100 orbits          */

    for (int i = 0; i < N; i++)
        step_verlet(&sc, &earth, 1, dt);

    double E1     = orbital_energy(&sc, &earth, 1);
    double rel_err = fabs(E1 - E0) / fabs(E0);

    /* Velocity-Verlet conserves energy to O(dt²) per step.
     * Over 50 000 steps we expect drift < 0.05 %. */
    ASSERT_NEAR(rel_err, 0.0, 5e-4,
                "energy drift < 0.05 % over 100 circular orbits");

    printf("         (E₀ = %.6e, E₁ = %.6e, rel err = %.2e)\n",
           E0, E1, rel_err);

    /*
     * After N complete orbits the craft should return close to its
     * starting RADIUS (not the exact Cartesian point, since discrete
     * timesteps won't land at exactly the same angle).
     */
    double r_final = v2_len(sc.pos);
    ASSERT_NEAR(fabs(r_final - r) / r, 0.0, 0.001,
                "radial drift < 0.1 % after 100 orbits");
}

/* ── 7. Collision detection ──────────────────────────────────── */
static void test_collision(void)
{
    test_section("Collision detection");

    Body bodies[2] = { make_earth(), make_moon() };

    /* Inside Earth radius → collision with index 0 */
    Spacecraft sc_in_earth;
    sc_in_earth.pos = v2(EARTH_RADIUS * 0.5, 0);
    sc_in_earth.vel = v2(0, 0);
    sc_in_earth.acc = v2(0, 0);
    sc_in_earth.time = 0; sc_in_earth.distance_traveled = 0;

    ASSERT_TRUE(check_collision(&sc_in_earth, bodies, 2) == 0,
                "craft inside Earth → collision index 0");

    /* On Moon's surface → collision with index 1 */
    Spacecraft sc_on_moon;
    sc_on_moon.pos = v2(EARTH_MOON_DIST + MOON_RADIUS * 0.9, 0);
    sc_on_moon.vel = v2(0, 0);
    sc_on_moon.acc = v2(0, 0);
    sc_on_moon.time = 0; sc_on_moon.distance_traveled = 0;

    ASSERT_TRUE(check_collision(&sc_on_moon, bodies, 2) == 1,
                "craft inside Moon → collision index 1");

    /* Between the two bodies → no collision */
    Spacecraft sc_free;
    sc_free.pos = v2(EARTH_MOON_DIST / 2.0, 0);
    sc_free.vel = v2(0, 0);
    sc_free.acc = v2(0, 0);
    sc_free.time = 0; sc_free.distance_traveled = 0;

    ASSERT_TRUE(check_collision(&sc_free, bodies, 2) == -1,
                "craft between bodies → no collision (-1)");
}

/* ── 8. N-body scenarios load correctly ──────────────────────── */
static void test_scenarios(void)
{
    test_section("Scenario loading");

    const Scenario *sc = get_scenarios();

    for (int i = 0; i < N_SCENARIOS; i++) {
        ASSERT_TRUE(sc[i].n >= 2 && sc[i].n <= MAX_BODIES,
                    sc[i].name);
        /* All bodies should have positive mass and radius */
        for (int b = 0; b < sc[i].n; b++) {
            ASSERT_TRUE(sc[i].bodies[b].mass   > 0.0, sc[i].bodies[b].name);
            ASSERT_TRUE(sc[i].bodies[b].radius > 0.0, sc[i].bodies[b].name);
        }
    }

    /* Earth-Moon: Moon should be at exactly EARTH_MOON_DIST from Earth */
    ASSERT_NEAR(v2_dist(sc[0].bodies[0].pos, sc[0].bodies[1].pos),
                EARTH_MOON_DIST, 1e-9,
                "Earth-Moon distance in scenario 0");

    /* Earth-Moon-Sun: Sun should be ~150 million km from Earth */
    ASSERT_NEAR(v2_dist(sc[1].bodies[0].pos, sc[1].bodies[2].pos),
                149600000.0, 1e-9,
                "Earth-Sun distance in scenario 1");
}

/* ── 9. Verlet vs Euler: energy comparison ───────────────────── */
static void test_verlet_vs_euler(void)
{
    test_section("Verlet vs Euler: energy stability comparison");

    Body earth = make_earth();
    double r   = 20000.0;
    double vc  = sqrt(G_CONST * EARTH_MASS / r);
    double T   = 2.0 * M_PI * r / vc;
    double dt  = T / 200.0;
    int    N   = 10 * 200;   /* 10 orbits */

    /* ── Velocity-Verlet ── */
    Spacecraft sv;
    sv.pos = v2(r, 0); sv.vel = v2(0, vc);
    sv.acc = grav_acc(&earth, sv.pos);
    sv.time = sv.distance_traveled = 0;
    double E0v = orbital_energy(&sv, &earth, 1);
    for (int i = 0; i < N; i++) step_verlet(&sv, &earth, 1, dt);
    double errV = fabs(orbital_energy(&sv, &earth, 1) - E0v) / fabs(E0v);

    /* ── Euler (manual, for comparison only) ── */
    double ex = r, ey = 0, evx = 0, evy = vc;
    Vec2 ea = grav_acc(&earth, v2(ex, ey));
    double E0e = 0.5*(evx*evx+evy*evy) - G_CONST*EARTH_MASS/sqrt(ex*ex+ey*ey);
    for (int i = 0; i < N; i++) {
        ea   = grav_acc(&earth, v2(ex, ey));
        evx += ea.x * dt; evy += ea.y * dt;
        ex  += evx  * dt; ey  += evy  * dt;
    }
    double E1e = 0.5*(evx*evx+evy*evy) - G_CONST*EARTH_MASS/sqrt(ex*ex+ey*ey);
    double errE = fabs(E1e - E0e) / fabs(E0e);

    printf("         Verlet energy drift: %.2e | Euler energy drift: %.2e\n",
           errV, errE);
    ASSERT_TRUE(errV < errE,
                "Verlet has lower energy drift than Euler over 10 orbits");
    ASSERT_NEAR(errV, 0.0, 0.01,
                "Verlet drift < 1 % over 10 orbits");
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

int main(void)
{
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║      Spacecraft Simulator — Physics Test Suite      ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    test_vec2();
    test_grav_acc();
    test_resultant_acc();
    test_escape_velocity();
    test_lagrange_l1();
    test_verlet_energy();
    test_collision();
    test_scenarios();
    test_verlet_vs_euler();

    printf("\n────────────────────────────────────────────────────\n");
    printf("  Result:  %s%d passed%s  /  %s%d failed%s  "
           "/ %d total\n",
           GREEN, _pass, RESET,
           _fail ? RED : GREEN, _fail, RESET,
           _pass + _fail);
    printf("────────────────────────────────────────────────────\n\n");

    return _fail;   /* 0 = all good, non-zero = failures */
}
