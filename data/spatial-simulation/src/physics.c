/*
 * physics.c — Gravitational physics and numerical integration
 */

#include "physics.h"

/*
 * Gravitational acceleration at `pos` due to `body`.
 *
 *   a⃗ = G·M · r⃗ / |r⃗|³
 *
 * where r⃗ = body.pos − pos  (points toward the attracting body).
 * Dividing by |r⃗|³ instead of |r⃗|² folds in the unit-vector normalisation.
 */
Vec2 grav_acc(const Body *body, Vec2 pos)
{
    Vec2   r    = v2_sub(body->pos, pos);
    double dist = v2_len(r);
    if (dist < EPSILON) return v2(0, 0);          /* avoid singularity */
    double mag  = G_CONST * body->mass / (dist * dist * dist);
    return v2_scale(r, mag);
}

/* Superposition principle: sum contributions from every body. */
Vec2 resultant_acc(const Body *bodies, int n, Vec2 pos)
{
    Vec2 total = v2(0, 0);
    for (int i = 0; i < n; i++)
        total = v2_add(total, grav_acc(&bodies[i], pos));
    return total;
}

/* Escape velocity: v_esc = √(2GM/r) */
double escape_velocity(const Body *body, Vec2 pos)
{
    double r = v2_dist(body->pos, pos);
    if (r < EPSILON) return 0.0;
    return sqrt(2.0 * G_CONST * body->mass / r);
}

/*
 * L1 Lagrange point (Hill-sphere approximation).
 *
 * The exact L1 lies on line a→b.  Distance from the larger body a:
 *
 *   d_a ≈ D · (1 − ∛(m_b / (3·m_a)))
 *
 * Valid when m_a >> m_b  (e.g. Earth-Moon, Sun-Earth).
 */
Vec2 lagrange_l1(const Body *a, const Body *b)
{
    double d   = v2_dist(a->pos, b->pos);
    double d_a = d * (1.0 - cbrt(b->mass / (3.0 * a->mass)));
    Vec2   dir = v2_norm(v2_sub(b->pos, a->pos));
    return v2_add(a->pos, v2_scale(dir, d_a));
}

/*
 * Velocity-Verlet (symplectic) integration step.
 *
 * Unlike Euler, this scheme is time-reversible and conserves energy to
 * O(dt²), making it far more suitable for long orbital simulations.
 *
 *   1. x(t+dt) = x(t) + v(t)·dt + ½·a(t)·dt²
 *   2. a(t+dt) = F(x(t+dt))           ← one force evaluation per step
 *   3. v(t+dt) = v(t) + ½·(a(t)+a(t+dt))·dt
 */
void step_verlet(Spacecraft *sc, const Body *bodies, int n, double dt)
{
    Vec2 a0 = sc->acc;
    Vec2 p0 = sc->pos;

    /* Step 1 — update position */
    sc->pos = v2_add(
        v2_add(sc->pos, v2_scale(sc->vel, dt)),
        v2_scale(a0, 0.5 * dt * dt)
    );

    /* Step 2 — evaluate acceleration at new position */
    Vec2 a1 = resultant_acc(bodies, n, sc->pos);

    /* Step 3 — update velocity with average acceleration */
    sc->vel = v2_add(sc->vel, v2_scale(v2_add(a0, a1), 0.5 * dt));
    sc->acc = a1;

    sc->distance_traveled += v2_dist(p0, sc->pos);
    sc->time              += dt;
}

/*
 * Velocity-Verlet integration for all n bodies (mutual n-body gravity).
 * Each body integrates independently under the superposition of all others.
 */
void step_bodies(Body *bodies, int n, double dt)
{
    if (n < 2) return;   /* nothing to integrate with only one body */

    /* Fixed-size stack arrays; n is always <= MAX_BODIES (8) */
    Vec2 acc0[8];
    Vec2 acc1[8];

    /* Step 1 — accelerations at current positions */
    for (int i = 0; i < n; i++) {
        acc0[i] = v2(0, 0);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            acc0[i] = v2_add(acc0[i], grav_acc(&bodies[j], bodies[i].pos));
        }
    }

    /* Step 2 — update positions */
    for (int i = 0; i < n; i++) {
        bodies[i].pos = v2_add(
            v2_add(bodies[i].pos, v2_scale(bodies[i].vel, dt)),
            v2_scale(acc0[i], 0.5 * dt * dt)
        );
    }

    /* Step 3 — accelerations at new positions */
    for (int i = 0; i < n; i++) {
        acc1[i] = v2(0, 0);
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            acc1[i] = v2_add(acc1[i], grav_acc(&bodies[j], bodies[i].pos));
        }
    }

    /* Step 4 — update velocities */
    for (int i = 0; i < n; i++) {
        bodies[i].vel = v2_add(
            bodies[i].vel,
            v2_scale(v2_add(acc0[i], acc1[i]), 0.5 * dt)
        );
    }
}

/*
 * Transform a position from the inertial frame into the co-rotating frame:
 *   1. Translate: subtract bodies[0].pos  (body[0] becomes origin)
 *   2. Rotate by -θ: θ = angle of bodies[1] from bodies[0]  (body[1] stays on +x)
 * If n < 2, only step 1 is applied (no rotation axis defined).
 */
Vec2 to_rotating_frame(Vec2 pos, const Body *bodies, int n)
{
    if (n < 1) return pos;

    /* Translate: origin at primary body */
    Vec2 p = v2_sub(pos, bodies[0].pos);
    if (n < 2) return p;

    /* Rotation axis: direction from bodies[0] to bodies[1] */
    Vec2   dir = v2_sub(bodies[1].pos, bodies[0].pos);
    double len = v2_len(dir);
    if (len < EPSILON) return p;

    double cx = dir.x / len;   /* cos θ */
    double sx = dir.y / len;   /* sin θ */

    /* Rotate by -θ: (x cosθ + y sinθ,  −x sinθ + y cosθ) */
    return v2(p.x * cx + p.y * sx,
             -p.x * sx + p.y * cx);
}

/* Returns the index of the first body the spacecraft overlaps, or -1. */
int check_collision(const Spacecraft *sc, const Body *bodies, int n)
{
    for (int i = 0; i < n; i++) {
        if (v2_dist(sc->pos, bodies[i].pos) < bodies[i].radius)
            return i;
    }
    return -1;
}
