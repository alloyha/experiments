/*
 * missions.h — Historical and fictional mission presets
 *
 * Each Mission binds a scenario index to a specific set of
 * initial spacecraft conditions with real-world-inspired numbers.
 */

#ifndef MISSIONS_H
#define MISSIONS_H

#define N_MISSIONS 7

typedef struct {
    const char *name;
    const char *year;
    const char *description;         /* two-line summary          */
    int         scenario_idx;        /* index into get_scenarios() */

    /* Initial spacecraft state (scenario coordinate frame, km / km·h⁻¹) */
    double x0, y0;
    double vx0, vy0;

    /* Recommended simulation parameters */
    double dt;
    double max_time;    /* h */
    int    steps_frame; /* steps per render frame */
} Mission;

/* Returns pointer to the static array of N_MISSIONS missions */
const Mission *get_missions(void);

#endif /* MISSIONS_H */
