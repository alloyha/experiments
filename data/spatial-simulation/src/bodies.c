/*
 * bodies.c — Predefined celestial bodies and simulation scenarios
 *
 * Unit system: km, kg, h  (same as physics.h)
 *
 * All masses and radii are from NASA/JPL fact sheets.
 * Distances shown are mean values at the epoch used.
 */

#include "bodies.h"
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Raw body parameters
 * ═══════════════════════════════════════════════════════════════ */

/* Sun */
#define SUN_MASS      1.989e30      /* kg          */
#define SUN_RADIUS    695700.0      /* km          */

/* Earth */
#define EARTH_MASS_   5.972e24      /* kg          */
#define EARTH_RADIUS_ 6371.0        /* km (mean)   */

/* Moon */
#define MOON_MASS_    7.342e22      /* kg          */
#define MOON_RADIUS_  1737.4        /* km          */
#define EARTH_MOON_D  384400.0      /* km, mean    */

/* Mars */
#define MARS_MASS     6.417e23      /* kg          */
#define MARS_RADIUS   3389.5        /* km          */

/* Phobos */
#define PHOBOS_MASS   1.066e16      /* kg          */
#define PHOBOS_RADIUS 11.267        /* km (mean)   */
#define MARS_PHOBOS_D 9376.0        /* km, semi-major */

/* Jupiter */
#define JUP_MASS      1.898e27      /* kg          */
#define JUP_RADIUS    69911.0       /* km          */

/* Io */
#define IO_MASS       8.932e22      /* kg          */
#define IO_RADIUS     1821.6        /* km          */
#define JUP_IO_D      421800.0      /* km, semi-major */

/* ── Initial orbital velocities (km/h, counterclockwise orbits) ────
 *
 * Computed from v = sqrt(G·M_primary / r) with G = 8.65e-13 km³ kg⁻¹ h⁻².
 *
 *   Moon  around Earth  : sqrt(8.65e-13 * 5.972e24 / 384400)   ≈ 3666 km/h
 *   Earth recoil (Moon) : 3666 * M_moon/M_earth                ≈ 45.1 km/h
 *   Earth around Sun    : sqrt(8.65e-13 * 1.989e30 / 149600000) ≈ 107238 km/h
 *   Phobos around Mars  : sqrt(8.65e-13 * 6.417e23 / 9376)      ≈ 7695 km/h
 *   Io around Jupiter   : sqrt(8.65e-13 * 1.898e27 / 421800)    ≈ 62378 km/h
 *
 * In every scenario bodies[0] is the "primary" (at origin), so the
 * secondary starts at (+distance, 0) with velocity in +y (CCW orbit).
 * In the Earth-Moon-Sun scenario Earth also carries its solar orbital
 * velocity, so Moon's absolute velocity = solar + cislunar components.
 * ─────────────────────────────────────────────────────────────────── */
#define MOON_ORB_VEL   3666.0       /* Moon orbital speed around Earth      */
#define EARTH_RECOIL   45.1         /* Earth counter-motion (−y) from Moon  */
#define EARTH_SUN_VEL  107238.0     /* Earth orbital speed around Sun (+y)  */
#define MOON_SUN_VEL   (EARTH_SUN_VEL + MOON_ORB_VEL)  /* Moon absolute    */
#define PHOBOS_ORB_VEL 7695.0       /* Phobos orbital speed around Mars     */
#define IO_ORB_VEL     62378.0      /* Io orbital speed around Jupiter      */

/* ═══════════════════════════════════════════════════════════════
 * Static scenario table
 * ═══════════════════════════════════════════════════════════════ */

static const Scenario SCENARIOS[N_SCENARIOS] = {

    /* ── 0: Earth-Moon ──────────────────────────────────────── */
    {
        .name        = "Earth-Moon",
        .description = "Classic two-body cislunar system.",
        .n           = 2,
        .bodies = {
            {
                .pos    = {0.0, 0.0},
                .vel    = {0.0, -EARTH_RECOIL},   /* counter-motion from Moon */
                .mass   = EARTH_MASS_,
                .radius = EARTH_RADIUS_,
                .name   = "Earth",
                .symbol = 'E',
            },
            {
                .pos    = {EARTH_MOON_D, 0.0},
                .vel    = {0.0, MOON_ORB_VEL},    /* CCW orbit around Earth */
                .mass   = MOON_MASS_,
                .radius = MOON_RADIUS_,
                .name   = "Moon",
                .symbol = 'L',
            },
        },
        .view_cx   = EARTH_MOON_D / 2.0,
        .view_cy   = 0.0,
        .view_span = EARTH_MOON_D * 1.25,
    },

    /* ── 1: Earth-Moon-Sun (solar perturbation) ─────────────── */
    {
        .name        = "Earth-Moon-Sun",
        .description = "Cislunar + solar gravity (long sims drift noticeably).",
        .n           = 3,
        .bodies = {
            {
                .pos    = {0.0, 0.0},
                .vel    = {0.0, EARTH_SUN_VEL},   /* CCW orbit around Sun */
                .mass   = EARTH_MASS_,
                .radius = EARTH_RADIUS_,
                .name   = "Earth",
                .symbol = 'E',
            },
            {
                .pos    = {EARTH_MOON_D, 0.0},
                .vel    = {0.0, MOON_SUN_VEL},    /* solar + cislunar velocity */
                .mass   = MOON_MASS_,
                .radius = MOON_RADIUS_,
                .name   = "Moon",
                .symbol = 'L',
            },
            {
                /* Sun is 149.6 million km from Earth, on the −x side */
                .pos    = {-149600000.0, 0.0},
                .vel    = {0.0, 0.0},             /* fixed stellar reference */
                .mass   = SUN_MASS,
                .radius = SUN_RADIUS,
                .name   = "Sun",
                .symbol = 'S',
            },
        },
        /* Viewport centres on Earth-Moon segment; Sun is off-screen */
        .view_cx   = EARTH_MOON_D / 2.0,
        .view_cy   = 0.0,
        .view_span = EARTH_MOON_D * 1.25,
    },

    /* ── 2: Mars-Phobos ─────────────────────────────────────── */
    {
        .name        = "Mars-Phobos",
        .description = "Low-gravity system. Phobos orbits in just 7.6 h.",
        .n           = 2,
        .bodies = {
            {
                .pos    = {0.0, 0.0},
                .vel    = {0.0, 0.0},             /* primary, kept fixed */
                .mass   = MARS_MASS,
                .radius = MARS_RADIUS,
                .name   = "Mars",
                .symbol = 'M',
            },
            {
                .pos    = {MARS_PHOBOS_D, 0.0},
                .vel    = {0.0, PHOBOS_ORB_VEL},  /* CCW orbit around Mars */
                .mass   = PHOBOS_MASS,
                .radius = PHOBOS_RADIUS,
                .name   = "Phobos",
                .symbol = 'P',
            },
        },
        .view_cx   = MARS_PHOBOS_D / 2.0,
        .view_cy   = 0.0,
        .view_span = MARS_PHOBOS_D * 2.5,
    },

    /* ── 3: Jupiter-Io ──────────────────────────────────────── */
    {
        .name        = "Jupiter-Io",
        .description = "Massive primary; Io orbits in ~42 h.",
        .n           = 2,
        .bodies = {
            {
                .pos    = {0.0, 0.0},
                .vel    = {0.0, 0.0},             /* primary, kept fixed */
                .mass   = JUP_MASS,
                .radius = JUP_RADIUS,
                .name   = "Jupiter",
                .symbol = 'J',
            },
            {
                .pos    = {JUP_IO_D, 0.0},
                .vel    = {0.0, IO_ORB_VEL},      /* CCW orbit around Jupiter */
                .mass   = IO_MASS,
                .radius = IO_RADIUS,
                .name   = "Io",
                .symbol = 'I',
            },
        },
        .view_cx   = JUP_IO_D / 2.0,
        .view_cy   = 0.0,
        .view_span = JUP_IO_D * 1.3,
    },
};

/* ── Public accessors ────────────────────────────────────────── */

const Scenario *get_scenarios(void) { return SCENARIOS; }

int scenario_load_bodies(const Scenario *sc,
                         Body dst_bodies[MAX_BODIES])
{
    memcpy(dst_bodies, sc->bodies, (size_t)sc->n * sizeof(Body));
    return sc->n;
}
