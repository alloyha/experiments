/*
 * physics.h — Types and physics function declarations
 *
 * Unit system throughout:
 *   distance  → km
 *   mass      → kg
 *   time      → h (hours)
 *
 * G [km³ kg⁻¹ h⁻²]:
 *   G_SI = 6.674e-11 m³/(kg·s²)
 *   × (1e-9 km³/m³) × (3600² s²/h²) ≈ 8.65e-13   ✓ (matches original)
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include <math.h>

/* ── Physical constants ─────────────────────────────────────── */

#define G_CONST          8.65e-13    /* km³ kg⁻¹ h⁻²           */
#define EARTH_MASS       5.97e24     /* kg                      */
#define MOON_MASS        7.35e22     /* kg                      */
#define EARTH_RADIUS     6378.0      /* km                      */
#define MOON_RADIUS      1738.0      /* km                      */
#define EARTH_MOON_DIST  384400.0    /* km  (centre-to-centre)  */
#define EPSILON          1e-9        /* numerical zero guard    */

/* ── 2-D vector ─────────────────────────────────────────────── */

typedef struct { double x, y; } Vec2;

/* All inline so the compiler can eliminate call overhead */
static inline Vec2   v2(double x, double y)     { return (Vec2){x, y};              }
static inline Vec2   v2_add(Vec2 a, Vec2 b)     { return (Vec2){a.x+b.x, a.y+b.y}; }
static inline Vec2   v2_sub(Vec2 a, Vec2 b)     { return (Vec2){a.x-b.x, a.y-b.y}; }
static inline Vec2   v2_scale(Vec2 v, double s) { return (Vec2){v.x*s, v.y*s};      }
static inline double v2_len(Vec2 v)             { return sqrt(v.x*v.x + v.y*v.y);   }
static inline double v2_dist(Vec2 a, Vec2 b)    { return v2_len(v2_sub(b, a));       }
static inline Vec2   v2_norm(Vec2 v) {
    double l = v2_len(v);
    return l > EPSILON ? v2_scale(v, 1.0 / l) : v2(0, 0);
}

/* ── Celestial body ──────────────────────────────────────────── */

typedef struct {
    Vec2        pos;     /* km            */
    Vec2        vel;     /* km/h          */
    double      mass;    /* kg            */
    double      radius;  /* km            */
    const char *name;    /* display label */
    char        symbol;  /* map glyph     */
} Body;

/* ── Spacecraft ─────────────────────────────────────────────── */

typedef struct {
    Vec2   pos;               /* km      */
    Vec2   vel;               /* km/h    */
    Vec2   acc;               /* km/h²   */
    double distance_traveled; /* km      */
    double time;              /* h       */
} Spacecraft;

/* ── Function prototypes ────────────────────────────────────── */

/* Gravitational acceleration at pos due to one body */
Vec2   grav_acc(const Body *body, Vec2 pos);

/* Superposition from n bodies */
Vec2   resultant_acc(const Body *bodies, int n, Vec2 pos);

/* Escape velocity: v = sqrt(2GM/r) */
double escape_velocity(const Body *body, Vec2 pos);

/*
 * L1 Lagrange point via Hill-sphere approximation:
 *   d_L1 ≈ D · (1 − ∛(m_b / 3·m_a))
 * where D = distance(a,b).  Accurate when m_a >> m_b.
 */
Vec2   lagrange_l1(const Body *a, const Body *b);

/*
 * Velocity-Verlet integration step (2nd-order, energy-conserving):
 *   x(t+dt) = x(t) + v(t)·dt + ½·a(t)·dt²
 *   a(t+dt) = F(x(t+dt))
 *   v(t+dt) = v(t) + ½·(a(t) + a(t+dt))·dt
 */
void   step_verlet(Spacecraft *sc, const Body *bodies, int n, double dt);

/* Velocity-Verlet integration for all n bodies (mutual n-body gravity).
 * Each body's vel must be initialised to its orbital velocity. */
void   step_bodies(Body *bodies, int n, double dt);

/* Transform position from inertial frame to co-rotating reference frame.
 * Origin = bodies[0].pos; x-axis = direction bodies[0]→bodies[1].
 * If n < 2, only translates (no rotation). */
Vec2   to_rotating_frame(Vec2 pos, const Body *bodies, int n);

/* Returns index of collided body, or -1 */
int    check_collision(const Spacecraft *sc, const Body *bodies, int n);

#endif /* PHYSICS_H */
