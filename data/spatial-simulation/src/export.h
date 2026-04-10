/*
 * export.h — Trajectory data export (CSV + gnuplot script + C animator)
 *
 * Usage:
 *   1. exporter_init(&ex, "my_mission");
 *   2. During sim: exporter_record(&ex, &sc, bodies, n);
 *   3. At the end: exporter_write(&ex);          writes .csv + .plt + _animate.c
 *   4. Optionally: exporter_render(&ex, flags);  runs gnuplot / compiles video
 *
 * Files created:
 *   <stem>.csv           — time-series data, one row per recorded step
 *   <stem>.plt           — self-contained gnuplot script (static plots)
 *   <stem>_animate.c     — self-contained C renderer (compile + run → .mp4)
 *
 * Render flags (combine with |):
 *   EXPORT_IMAGES  — run gnuplot to produce trajectory.png + telemetry.png
 *   EXPORT_VIDEO   — compile _animate.c and pipe frames to ffmpeg → .mp4
 */

#ifndef EXPORT_H
#define EXPORT_H

#include "physics.h"

/* Render flags for exporter_render() — combine with | */
#define EXPORT_IMAGES  (1 << 0)   /* gnuplot → trajectory.png + telemetry.png */
#define EXPORT_VIDEO   (1 << 1)   /* compile _animate.c + ffmpeg → .mp4       */

#define EXPORT_STEM_MAX  64
#define EXPORT_MAX_PTS   500000   /* ~500 k rows ≈ few MB        */

typedef struct {
    /* Configuration */
    char   stem[EXPORT_STEM_MAX]; /* base filename (no extension) */
    int    decimation;            /* record every N-th step       */

    /* Accumulated trajectory */
    double *time;
    double *px, *py;
    double *vx, *vy;
    double *speed;
    double *dist[8];              /* distance to each body        */
    int     n_bodies;
    char    body_names[8][32];
    /* Body positions: initial snapshot + history in rotating frame */
    double  body_px[8];           /* body x position (initial)     */
    double  body_py[8];           /* body y position (initial)     */
    double *body_px_hist[8];      /* body x position history       */
    double *body_py_hist[8];      /* body y position history       */
    double  body_mass[8];         /* body mass (kg) */
    double  body_radius[8];       /* body radius (km) for collision detection */

    int    count;        /* rows recorded so far */
    int    cap;          /* allocated capacity   */
    int    step_counter; /* decimation counter   */
    double dt_sim;       /* simulation timestep (h) — used for phase portrait */

    /* Mission metadata written into the plot header */
    const char *mission_name;
    const char *scenario_name;
} Exporter;

/* Initialise; stem must be a valid filename base (no slashes).
 * decimation = 1 → every step; 10 → every 10th step, etc. */
void exporter_init(Exporter *ex,
                   const char *stem,
                   int         decimation,
                   double      dt_sim,
                   const char *mission_name,
                   const char *scenario_name);

/* Record one data point (call after each sim step). */
void exporter_record(Exporter          *ex,
                     const Spacecraft  *sc,
                     const Body        *bodies,
                     int                n_bodies);

/* Write .csv, .plt, and _animate.c files.  Returns 0 on success, -1 on error. */
int exporter_write(const Exporter *ex);

/* Run post-processing (gnuplot images and/or animation) after exporter_write().
 * flags: EXPORT_IMAGES | EXPORT_VIDEO (or 0 for nothing).
 * Returns bitmask of successfully completed steps. */
int exporter_render(const Exporter *ex, int flags);

/* Free internal buffers. */
void exporter_free(Exporter *ex);

#endif /* EXPORT_H */
