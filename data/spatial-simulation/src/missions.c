/*
 * missions.c — Mission presets with historically-inspired parameters
 *
 * Numbers are derived from NASA mission reports, converted to
 * the simulator's unit system (km, h).
 *
 * Reference velocities in km/h = km/s × 3600.
 *
 * EARTH RADIUS = 6371 km  (mean)
 * LEO altitude ≈ 185–200 km  →  r ≈ 6556–6571 km from Earth centre
 *
 * Trans-Lunar Injection (TLI) burn raises apogee to ≈ 400 000 km.
 * After TLI the spacecraft velocity at perigee ≈ 10.82 km/s = 38 952 km/h.
 */

#include "missions.h"

/* ── Earth-Moon geometry ─────────────────────────────────────── */
#define EARTH_R   6371.0    /* km, mean radius            */
#define LEO_ALT   185.0     /* km, low parking orbit alt  */
#define LEO_R     (EARTH_R + LEO_ALT)   /* 6556 km        */

/* ── Trans-Lunar Injection velocities ───────────────────────── */
/*
 * TLI velocity at perigee (km/h):
 *   Apollo 11  : 10.844 km/s  → 39 038 km/h
 *   Artemis I  : 10.210 km/s  → 36 756 km/h  (SLS performance margin)
 *
 * Spacecraft starts on the +x axis (to the right of Earth),
 * heading in the +y direction (counter-clockwise, same sense as
 * the Moon's orbit).
 */

static const Mission MISSIONS[N_MISSIONS] = {

    /* ── 0: Apollo 11 TLI ────────────────────────────────────── */
    /*
     * Moon starts at (+384400, 0).  To do a Hohmann transfer toward
     * the Moon the spacecraft must start on the opposite side of Earth
     * (−x) so that the apogee of the transfer ellipse falls at +x.
     * Initial position: (−LEO_R, 0);  velocity: (0, +TLI) — prograde.
     */
    {
        .name         = "Apollo 11",
        .year         = "1969",
        .description  = "Trans-Lunar Injection from 185 km parking orbit.\n"
                        "  First crewed lunar landing (Sea of Tranquility).",
        .scenario_idx = 0,   /* Earth-Moon */
        .x0           = -LEO_R,
        .y0           = 0.0,
        .vx0          = 0.0,
        .vy0          = 39038.0,   /* 10.844 km/s */
        .dt           = 0.02,
        .max_time     = 120.0,
        .steps_frame  = 150,
    },

    /* ── 1: Apollo 13 ─────────────────────────────────────────── */
    {
        .name         = "Apollo 13",
        .year         = "1970",
        .description  = "Free-return trajectory used after oxygen tank failure.\n"
                        "  Spacecraft slingshots around Moon and returns safely.",
        .scenario_idx = 0,
        .x0           = -LEO_R,
        .y0           = 0.0,
        .vx0          = 0.0,
        .vy0          = 38650.0,   /* ~10.74 km/s */
        .dt           = 0.02,
        .max_time     = 200.0,
        .steps_frame  = 150,
    },

    /* ── 2: Artemis I (Orion, uncrewed) ──────────────────────── */
    {
        .name         = "Artemis I",
        .year         = "2022",
        .description  = "Orion capsule on SLS — distant retrograde orbit\n"
                        "  around Moon, then return to Earth.",
        .scenario_idx = 0,
        .x0           = -LEO_R,
        .y0           = 0.0,
        .vx0          = 0.0,
        .vy0          = 36756.0,   /* 10.21 km/s */
        .dt           = 0.02,
        .max_time     = 300.0,
        .steps_frame  = 200,
    },

    /* ── 3: Earth Escape ─────────────────────────────────────── */
    {
        .name         = "Earth Escape",
        .year         = "synthetic",
        .description  = "Velocity exceeds Earth escape speed (~11.2 km/s).\n"
                        "  Craft leaves the Earth-Moon system entirely.",
        .scenario_idx = 1,   /* Earth-Moon-Sun: solar gravity matters here */
        .x0           = -LEO_R,
        .y0           = 0.0,
        .vx0          = 0.0,
        .vy0          = 42000.0,   /* 11.67 km/s — above 11.2 km/s escape */
        .dt           = 0.05,
        .max_time     = 600.0,
        .steps_frame  = 300,
    },

    /* ── 4: Moon Impact ──────────────────────────────────────── */
    /*
     * Direct hyperbolic crash trajectory into the Moon.
     * Start at LEO altitude on the Earth-Moon line (+x side) so there
     * is a clear path to the Moon without passing through Earth.
     * Speed ~50 000 km/h (well above escape, ~13.9 km/s) reaches Moon
     * altitude in ≈ 9 h.  The +y lead (vy ≈ +4 500 km/h at ~5° off-axis)
     * compensates for the Moon moving counter-clockwise during transit.
     * Moon gravity focuses the approach into an impact.
     */
    {
        .name         = "Moon Impact",
        .year         = "synthetic",
        .description  = "Hyperbolic crash trajectory into the Moon.\n"
                        "  Launched from LEO toward the Moon's leading position.",
        .scenario_idx = 0,
        .x0           =  6556.0,    /* LEO altitude, Moon-facing side (+x) */
        .y0           =     0.0,
        .vx0          = 49800.0,    /* ~13.8 km/s — well above escape       */
        .vy0          =  4500.0,    /* ~5° lead to intercept moving Moon     */
        .dt           = 0.005,
        .max_time     = 12.0,
        .steps_frame  = 40,
    },

    /* ── 5: Mars Approach / Phobos Capture ───────────────────── */
    {
        .name         = "Mars Approach",
        .year         = "synthetic",
        .description  = "Spacecraft arrives at Mars and attempts to be\n"
                        "  captured into a Phobos-crossing orbit.",
        .scenario_idx = 2,   /* Mars-Phobos */
        /* Start 500 km above Mars surface, heading retrograde */
        .x0           = 3889.5,   /* Mars R + 500 km */
        .y0           = 0.0,
        .vx0          = 0.0,
        .vy0          = 12600.0,  /* ~3.5 km/s, slightly above circular */
        .dt           = 0.002,
        .max_time     = 30.0,
        .steps_frame  = 80,
    },

    /* ── 6: Jupiter Io Flyby ──────────────────────────────────── */
    {
        .name         = "Jupiter-Io Flyby",
        .year         = "synthetic",
        .description  = "Powered flyby through the Jupiter-Io system.\n"
                        "  Jupiter's gravity dominates; Io is a minor perturber.",
        .scenario_idx = 3,   /* Jupiter-Io */
        .x0           = 80000.0,
        .y0           = 200000.0,
        .vx0          = 80000.0,   /* 22.2 km/s inbound */
        .vy0          = -50000.0,
        .dt           = 0.001,
        .max_time     = 5.0,
        .steps_frame  = 100,
    },
};

const Mission *get_missions(void) { return MISSIONS; }
