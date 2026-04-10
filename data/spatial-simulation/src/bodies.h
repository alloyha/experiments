/*
 * bodies.h — Predefined celestial bodies and simulation scenarios
 *
 * A Scenario bundles a set of bodies, viewport hints, and a name.
 * Adding a new scenario requires only a new entry in bodies.c.
 */

#ifndef BODIES_H
#define BODIES_H

#include "physics.h"

#define MAX_BODIES   8
#define N_SCENARIOS  4

/* ── Scenario ─────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    const char *description;
    Body        bodies[MAX_BODIES];
    int         n;
    /* Viewport recommendation (world km) */
    double      view_cx;     /* centre x  */
    double      view_cy;     /* centre y  */
    double      view_span;   /* total x width shown */
} Scenario;

/* Returns pointer to the static array of N_SCENARIOS scenarios */
const Scenario *get_scenarios(void);

/* Deep-copies a scenario's body list into dst_bodies[MAX_BODIES].
 * Returns n_bodies copied. */
int scenario_load_bodies(const Scenario *sc,
                         Body dst_bodies[MAX_BODIES]);

#endif /* BODIES_H */
