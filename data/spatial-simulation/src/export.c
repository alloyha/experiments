/*
 * export.c — Trajectory export: CSV data + gnuplot script
 */

#include "export.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── init ────────────────────────────────────────────────────── */

void exporter_init(Exporter   *ex,
                   const char *stem,
                   int         decimation,
                   double      dt_sim,
                   const char *mission_name,
                   const char *scenario_name)
{
    memset(ex, 0, sizeof *ex);
    strncpy(ex->stem, stem, EXPORT_STEM_MAX - 1);
    ex->decimation    = (decimation < 1) ? 1 : decimation;
    ex->dt_sim        = (dt_sim > 0.0)   ? dt_sim : 0.01;
    ex->mission_name  = mission_name  ? mission_name  : "unknown";
    ex->scenario_name = scenario_name ? scenario_name : "unknown";

    /* Pre-allocate with a modest initial capacity */
    ex->cap  = 4096;
    ex->time  = malloc((size_t)ex->cap * sizeof(double));
    ex->px    = malloc((size_t)ex->cap * sizeof(double));
    ex->py    = malloc((size_t)ex->cap * sizeof(double));
    ex->vx    = malloc((size_t)ex->cap * sizeof(double));
    ex->vy    = malloc((size_t)ex->cap * sizeof(double));
    ex->speed = malloc((size_t)ex->cap * sizeof(double));
    for (int i = 0; i < 8; i++) {
        ex->dist[i] = malloc((size_t)ex->cap * sizeof(double));
        ex->body_px_hist[i] = malloc((size_t)ex->cap * sizeof(double));
        ex->body_py_hist[i] = malloc((size_t)ex->cap * sizeof(double));
    }
}

/* ── internal realloc helper ─────────────────────────────────── */

/* Returns 0 on success, -1 if any allocation failed. */
static int grow(Exporter *ex)
{
    int new_cap = ex->cap * 2;
    if (new_cap > EXPORT_MAX_PTS) new_cap = EXPORT_MAX_PTS;
    if (new_cap <= ex->cap) return -1;   /* hit the hard limit */

#define REALLOC(field) \
    do { \
        void *_p = realloc(ex->field, (size_t)new_cap * sizeof(double)); \
        if (!_p) return -1; \
        ex->field = _p; \
    } while(0)

    REALLOC(time);
    REALLOC(px); REALLOC(py);
    REALLOC(vx); REALLOC(vy);
    REALLOC(speed);
    for (int i = 0; i < 8; i++) {
        REALLOC(dist[i]);
        REALLOC(body_px_hist[i]);
        REALLOC(body_py_hist[i]);
    }
#undef REALLOC

    ex->cap = new_cap;
    return 0;
}

/* ── record ──────────────────────────────────────────────────── */

void exporter_record(Exporter         *ex,
                     const Spacecraft *sc,
                     const Body       *bodies,
                     int               n_bodies)
{
    if ((ex->step_counter++ % ex->decimation) != 0) return;
    if (ex->count >= EXPORT_MAX_PTS) return;
    if (ex->count >= ex->cap && grow(ex) < 0) return;

    int i = ex->count;
    ex->time[i]  = sc->time;

    /* Record position and velocity in the co-rotating reference frame:
     *   origin = bodies[0],  x-axis = direction bodies[0]→bodies[1].
     * Velocity is expressed relative to the reference body (bodies[0]).  */
    Vec2 pos_rf = to_rotating_frame(sc->pos, bodies, n_bodies);
    Vec2 vel_rf = (n_bodies > 0) ? v2_sub(sc->vel, bodies[0].vel) : sc->vel;
    ex->px[i]    = pos_rf.x;
    ex->py[i]    = pos_rf.y;
    ex->vx[i]    = vel_rf.x;
    ex->vy[i]    = vel_rf.y;
    ex->speed[i] = v2_len(vel_rf);

    int nb = n_bodies > 8 ? 8 : n_bodies;
    ex->n_bodies = nb;
    for (int b = 0; b < nb; b++) {
        ex->dist[b][i] = v2_dist(sc->pos, bodies[b].pos);
        strncpy(ex->body_names[b], bodies[b].name, 31);
        
        /* Transform body position to co-rotating frame at each timestep */
        Vec2 bp = to_rotating_frame(bodies[b].pos, bodies, n_bodies);
        ex->body_px_hist[b][i] = bp.x;
        ex->body_py_hist[b][i] = bp.y;
        
        /* Store initial body positions (snapshot for backward compatibility) */
        if (i == 0) {
            ex->body_px[b]     = bp.x;
            ex->body_py[b]     = bp.y;
            ex->body_mass[b]   = bodies[b].mass;
            ex->body_radius[b] = bodies[b].radius;
        }
    }

    ex->count++;
}

/* ── write CSV ───────────────────────────────────────────────── */

static int write_csv(const Exporter *ex, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;

    /* Header */
    fprintf(f, "# Spacecraft Trajectory — %s (%s)\n",
            ex->mission_name, ex->scenario_name);
    fprintf(f, "time_h,x_km,y_km,vx_kmh,vy_kmh,speed_kmh");
    for (int b = 0; b < ex->n_bodies; b++)
        fprintf(f, ",dist_%s_km", ex->body_names[b]);
    for (int b = 0; b < ex->n_bodies; b++)
        fprintf(f, ",bx_%s_km,by_%s_km", ex->body_names[b], ex->body_names[b]);
    fprintf(f, "\n");

    /* Data rows */
    for (int i = 0; i < ex->count; i++) {
        fprintf(f, "%.6f,%.3f,%.3f,%.4f,%.4f,%.4f",
                ex->time[i],
                ex->px[i], ex->py[i],
                ex->vx[i], ex->vy[i],
                ex->speed[i]);
        for (int b = 0; b < ex->n_bodies; b++)
            fprintf(f, ",%.3f", ex->dist[b][i]);
        for (int b = 0; b < ex->n_bodies; b++)
            fprintf(f, ",%.3f,%.3f", ex->body_px_hist[b][i], ex->body_py_hist[b][i]);
        fprintf(f, "\n");
    }

    fclose(f);
    return 0;
}

/* ── emit gnuplot xtics/ytics in ×10³ km with raw-km positions ─────── */
/*
 * Writes a gnuplot "set Xtics (...)" line where the tick POSITIONS are in
 * raw km but the LABELS show the value divided by 1000 (i.e., ×10³ km).
 * Picks ~5 evenly-spaced nice round ticks across [lo, hi].
 * axis == 'x' or 'y'.
 */
static void write_scaled_tics(FILE *f, char axis, double lo, double hi)
{
    double span = hi - lo;
    if (span <= 0.0) span = 1.0;

    /* Target exactly 10 intervals: rough step = span/10, then round up to
     * the nearest value in the {1, 2, 5} × 10^n family so tick labels
     * are nice round numbers.  This reliably produces 8–11 ticks. */
    double rough = span / 10.0;
    double mag   = pow(10.0, floor(log10(rough)));
    double step;
    if      (mag          >= rough) step = mag;
    else if (2.0 * mag    >= rough) step = 2.0 * mag;
    else if (5.0 * mag    >= rough) step = 5.0 * mag;
    else                            step = 10.0 * mag;

    double first = ceil(lo / step) * step;
    fprintf(f, "set %ctics (", axis);
    int count = 0;
    for (double t = first; t <= hi + step * 0.01; t += step) {
        if (count > 0) fprintf(f, ", ");
        /* Label = position / 1000, formatted without unnecessary decimals */
        double label_val = t / 1000.0;
        if (label_val == (long long)label_val)
            fprintf(f, "\"%.0f\" %g", label_val, t);
        else
            fprintf(f, "\"%g\" %g", label_val, t);
        count++;
    }
    fprintf(f, ")\n");
}

/* ── body marker colours (shared by trajectory + animation scripts) ─ */
static const char * const BODY_COLORS[8] = {
    "#00ffff",  /* cyan    — Earth / primary  */
    "#aaaaaa",  /* gray    — Moon             */
    "#ffff00",  /* yellow  — Sun              */
    "#00ff00",  /* green   — Body4            */
    "#ff8800",  /* orange                     */
    "#ff00ff",  /* magenta                    */
    "#ff4444",  /* red                        */
    "#8888ff",  /* light blue                 */
};

/* ── write gnuplot script ────────────────────────────────────── */

static int write_gnuplot(const Exporter *ex,
                         const char     *csv_path,
                         const char     *plt_path)
{
    FILE *f = fopen(plt_path, "w");
    if (!f) return -1;

    /* Determine bounding box from data */
    double xmin = ex->px[0], xmax = ex->px[0];
    double ymin = ex->py[0], ymax = ex->py[0];
    for (int i = 1; i < ex->count; i++) {
        if (ex->px[i] < xmin) xmin = ex->px[i];
        if (ex->px[i] > xmax) xmax = ex->px[i];
        if (ex->py[i] < ymin) ymin = ex->py[i];
        if (ex->py[i] > ymax) ymax = ex->py[i];
    }
    /* Trajectory-only bounds — used for phase-plane axes so distant bodies
     * (e.g., the Sun in an earth-escape mission) do not compress the axes */
    double xtmin = xmin, xtmax = xmax, ytmin = ymin, ytmax = ymax;
    double tmargin = (xtmax - xtmin) * 0.10 + 5000.0;

    /* Expand map viewport to include nearby bodies.
     * Bodies more than 4× the trajectory diagonal away are skipped so they
     * cannot compress the trajectory to an unreadable dot. */
    double traj_diag = sqrt((xtmax - xtmin) * (xtmax - xtmin) +
                            (ytmax - ytmin) * (ytmax - ytmin)) + 5000.0;
    for (int b = 0; b < ex->n_bodies; b++) {
        double bdx = ex->body_px[b] - (xtmin + xtmax) * 0.5;
        double bdy = ex->body_py[b] - (ytmin + ytmax) * 0.5;
        if (sqrt(bdx * bdx + bdy * bdy) > 4.0 * traj_diag) continue;
        if (ex->body_px[b] < xmin) xmin = ex->body_px[b];
        if (ex->body_px[b] > xmax) xmax = ex->body_px[b];
        if (ex->body_py[b] < ymin) ymin = ex->body_py[b];
        if (ex->body_py[b] > ymax) ymax = ex->body_py[b];
    }
    double margin = (xmax - xmin) * 0.08 + 5000.0;
    double speed_max = 0;
    for (int i = 0; i < ex->count; i++)
        if (ex->speed[i] > speed_max) speed_max = ex->speed[i];
    speed_max /= 1000.0;   /* convert to x1e3 km/h for colorbar display */

    fprintf(f,
        "# ─────────────────────────────────────────────────────\n"
        "# Gnuplot script — %s\n"
        "# Generated by Spacecraft Trajectory Simulator\n"
        "#\n"
        "# Run:  gnuplot %s\n"
        "#       (produces trajectory.png and telemetry.png)\n"
        "# ─────────────────────────────────────────────────────\n\n"
        "set datafile separator ','\n\n",
        ex->mission_name, plt_path);

    /* ── Figure 1: trajectory map (colour = speed) ── */
    /* Canvas width adapts to data aspect ratio so equal-aspect trajectories
     * fill the image rather than leaving large blank margins. */
    {
        double data_ar = (xmax + margin - (xmin - margin)) /
                         fmax(ymax + margin - (ymin - margin), 1.0);
        int c1w = (int)(900.0 * fmin(fmax(data_ar, 0.4), 2.7));
        int c1h = 900;
        if (c1w < 900) { c1h = (int)(900.0 / fmax(data_ar, 0.1)); c1w = 900; }
        if (c1w > 2400) c1w = 2400;
        if (c1h > 1200) c1h = 1200;
        fprintf(f, "set terminal pngcairo size %d,%d enhanced font 'Sans,11'\n",
                c1w, c1h);
    }
    fprintf(f,
        "set output 'trajectory_%s.png'\n\n"
        "set title \"%s \xe2\x80\x94 Trajectory Map\" font 'Sans Bold,14'\n"
        "set xlabel 'X  (\xc3\x97 10{^3} km)'\n"
        "set ylabel 'Y  (\xc3\x97 10{^3} km)'\n"
        "set size ratio -1\n"
        "set grid lc rgb '#333333' lw 0.4\n"
        "set colorbox\n"
        "set cbrange [0:%g]\n"
        "set cblabel 'Speed  (\xc3\x97 10{^3} km/h)'\n"
        "set palette defined (0 '#0000ff', 0.33 '#00ffff',\\\n"
        "                     0.66 '#ffff00', 1 '#ff0000')\n"
        "set xrange [%g:%g]\n"
        "set yrange [%g:%g]\n",
        ex->stem,
        ex->mission_name,
        speed_max,
        xmin - margin, xmax + margin,
        ymin - margin, ymax + margin);

    /* Custom tick labels that display ×10³ km values for raw-km data */
    write_scaled_tics(f, 'x', xmin - margin, xmax + margin);
    write_scaled_tics(f, 'y', ymin - margin, ymax + margin);
    fprintf(f, "\n");

    /* Body circles: show motion in rotating frame at start, middle, and end */
    double plot_span = (xmax - xmin + 2.0 * margin);
    double min_r     = plot_span * 0.002;   /* 0.2% of plot width, in km */
    
    for (int b = 0; b < ex->n_bodies; b++) {
        double r = ex->body_radius[b];
        if (r < min_r) r = min_r;
        
        /* Draw body at 3 points: start (opaque), middle (translucent), end (faint) */
        int start_idx  = 0;
        int mid_idx    = ex->count / 2;
        int end_idx    = ex->count - 1;
        if (end_idx < 0) end_idx = 0;
        
        /* Start: full opacity */
        fprintf(f,
            "set object %d circle at %g,%g size %g "
            "fc rgb '%s' fs solid 0.85 front\n",
            b + 1,
            ex->body_px_hist[b][start_idx], ex->body_py_hist[b][start_idx],
            r,
            BODY_COLORS[b < 8 ? b : 7]);
        
        /* Middle: medium opacity (object ID = n_bodies + b + 1) */
        if (mid_idx > start_idx) {
            fprintf(f,
                "set object %d circle at %g,%g size %g "
                "fc rgb '%s' fs solid 0.45 front\n",
                ex->n_bodies + b + 1,
                ex->body_px_hist[b][mid_idx], ex->body_py_hist[b][mid_idx],
                r,
                BODY_COLORS[b < 8 ? b : 7]);
        }
        
        /* End: low opacity (object ID = 2*n_bodies + b + 1) */
        if (end_idx > start_idx) {
            fprintf(f,
                "set object %d circle at %g,%g size %g "
                "fc rgb '%s' fs solid 0.20 front\n",
                2 * ex->n_bodies + b + 1,
                ex->body_px_hist[b][end_idx], ex->body_py_hist[b][end_idx],
                r,
                BODY_COLORS[b < 8 ? b : 7]);
        }
    }
    
    /* Body labels at current (most recent) position */
    for (int b = 0; b < ex->n_bodies; b++) {
        int end_idx = ex->count - 1;
        if (end_idx < 0) end_idx = 0;
        fprintf(f,
            "set label %d '%s' at %g,%g "
            "tc rgb '%s' front offset character 0.6,0.6\n",
            b + 1, ex->body_names[b],
            ex->body_px_hist[b][end_idx], ex->body_py_hist[b][end_idx],
            BODY_COLORS[b < 8 ? b : 7]);
    }
    fprintf(f, "\n");

    /* Time milestone marks (6 interior points, evenly spaced in record index).
     * Label numbers start after body labels to avoid collisions. */
    {
        int n_mile = 6;
        int lbase  = ex->n_bodies + 1;
        for (int k = 1; k <= n_mile; k++) {
            int    idx    = k * (ex->count - 1) / (n_mile + 1);
            double t_days = ex->time[idx] / 24.0;
            fprintf(f,
                "set label %d '%.1fd' at %g,%g "
                "point pt 7 ps 0.7 lc rgb '#cccccc' "
                "tc rgb '#444444' front offset character 0.4,0.6\n",
                lbase + k - 1, t_days,
                ex->px[idx], ex->py[idx]);
        }
    }
    fprintf(f, "\n");

    /* Trajectory as coloured line (speed ÷1000 → ×10³ km/h for palette mapping)
     * x/y use raw column numbers so lc palette renders correctly.
     * Body trajectories (dashed) are drawn first so they sit behind the
     * spacecraft path.  Column layout: cols 1-6 fixed, cols 7…6+N dist,
     * then bx/by pairs: bx_b = 7+N+2b, by_b = 7+N+2b+1  (1-based). */
    fprintf(f, "plot \\\n");
    for (int b = 0; b < ex->n_bodies; b++) {
        int bx_col = 7 + ex->n_bodies + 2 * b;
        int by_col = bx_col + 1;
        fprintf(f,
            "     '%s' using %d:%d with lines "
            "lw 1.5 lc rgb '%s' dt 2 title '%s',\\\n",
            csv_path, bx_col, by_col,
            BODY_COLORS[b < 8 ? b : 7],
            ex->body_names[b]);
    }
    fprintf(f,
        "     '%s' using 2:3:($6/1000) with lines \\\n"
        "         lw 1.5 lc palette title 'Trajectory',\\\n",
        csv_path);

    /* Start / end markers — raw km coordinates */
    fprintf(f,
        "     '%s' using ($0==1?$2:1/0):($0==1?$3:1/0) \\\n"
        "         with points pt 7 ps 2 lc rgb '#00ff00' title 'Start',\\\n"
        "     '%s' using ($0==%d?$2:1/0):($0==%d?$3:1/0) \\\n"
        "         with points pt 2 ps 2 lc rgb '#ff4444' title 'End'\n\n",
        csv_path,
        csv_path, ex->count, ex->count);

    /* Clean up body objects, body labels, and milestone labels before Figure 2 */
    for (int b = 0; b < ex->n_bodies; b++) {
        fprintf(f, "unset object %d\n", b + 1);           /* start */
        fprintf(f, "unset object %d\n", ex->n_bodies + b + 1);     /* middle */
        fprintf(f, "unset object %d\n", 2*ex->n_bodies + b + 1);   /* end */
        fprintf(f, "unset label %d\n", b + 1);
    }
    for (int k = 1; k <= 6; k++)
        fprintf(f, "unset label %d\n", ex->n_bodies + k);
    fprintf(f, "\n");

    /* ── Figure 2: telemetry over time (3 panels) ── */
    /* Reset custom tick labels set for Figure 1 so auto-ranging works. */
    fprintf(f,
        "set xtics autofreq\n"
        "set ytics autofreq\n"
        "set format x '%%g'\n"
        "set format y '%%g'\n\n");

    fprintf(f,
        "set terminal pngcairo size 1200,1100 enhanced font 'Sans,11'\n"
        "set output 'telemetry_%s.png'\n"
        "unset colorbox\n"
        "set size noratio\n"
        "set grid\n"
        "set xrange [*:*]\n"
        "set yrange [*:*]\n"
        "unset xlabel\n\n"
        "set multiplot layout 3,1 spacing 0.04 "
        "title \"%s \xe2\x80\x94 Telemetry\" font 'Sans Bold,14'\n\n",
        ex->stem, ex->mission_name);

    /* Panel 1: Speed (\xc3\x97 10\xc2\xb3 km/h) */
    fprintf(f,
        "set ylabel '\xc3\x97 10{^3} km/h'\n"
        "set title 'Speed'\n"
        "plot '%s' using ($1/24):($6/1000) with lines lw 2 lc rgb '#00aaff' notitle\n\n",
        csv_path);

    /* Panel 2: Velocity components vx and vy */
    fprintf(f,
        "set ylabel '\xc3\x97 10{^3} km/h'\n"
        "set title 'Velocity Components'\n"
        "set yrange [*:*]\n"
        "plot '%s' using ($1/24):($4/1000) with lines lw 2 lc rgb '#ff5555' title 'v_x',\\\n"
        "     '%s' using ($1/24):($5/1000) with lines lw 2 lc rgb '#33cc55' title 'v_y'\n\n",
        csv_path, csv_path);

    /* Panel 3: Distance to each body (\xc3\x97 10\xc2\xb3 km) */
    fprintf(f,
        "set xlabel 'Time  (days)'\n"
        "set ylabel '\xc3\x97 10{^3} km'\n"
        "set title 'Distance to Bodies'\n"
        "set yrange [*:*]\n"
        "plot ");
    for (int b = 0; b < ex->n_bodies; b++) {
        int col = 7 + b;   /* columns: 1=time, 7+ = distances */
        if (b > 0) fprintf(f, "     ");
        fprintf(f, "'%s' using ($1/24):($%d/1000) with lines lw 1.5 title '%s'%s\n",
                csv_path, col, ex->body_names[b],
                b < ex->n_bodies - 1 ? ",\\" : "");
    }

    fprintf(f, "\nunset multiplot\n\n");

    /* ── Figure 4: coordinate phase planes (x vs ẋ, y vs ẏ) ──────────
     *
     * 1×2 side-by-side layout:
     *   Left  — x  (×10³ km)  vs  vx (×10³ km/h)
     *   Right — y  (×10³ km)  vs  vy (×10³ km/h)
     * Both coloured by elapsed time (days) so the evolution direction is
     * immediately visible.
     * ──────────────────────────────────────────────────────────────── */
    {
        /* Gather ranges for x-vx and y-vy panels */
        double vxmin = ex->vx[0], vxmax = ex->vx[0];
        double vymin = ex->vy[0], vymax = ex->vy[0];
        for (int i = 1; i < ex->count; i++) {
            if (ex->vx[i] < vxmin) vxmin = ex->vx[i];
            if (ex->vx[i] > vxmax) vxmax = ex->vx[i];
            if (ex->vy[i] < vymin) vymin = ex->vy[i];
            if (ex->vy[i] > vymax) vymax = ex->vy[i];
        }
        double t_days_max = ex->time[ex->count - 1] / 24.0;

        double vx_pad = (vxmax - vxmin) * 0.06 + 200.0;
        double vy_pad = (vymax - vymin) * 0.06 + 200.0;

        fprintf(f,
            "set terminal pngcairo size 1400,700 enhanced font 'Sans,11'\n"
            "set output 'phase_planes_%s.png'\n"
            "set colorbox\n"
            "set cbrange [0:%g]\n"
            "set cblabel 'Time  (days)'\n"
            "set palette defined (0 '#0000ff', 0.33 '#00ffff',\\\n"
            "                     0.66 '#ffff00', 1 '#ff0000')\n"
            "set size noratio\n"
            "set grid\n"
            "set xzeroaxis lw 0.8 lt 0 lc rgb '#888888'\n"
            "set yzeroaxis lw 0.8 lt 0 lc rgb '#888888'\n"
            "set multiplot layout 1,2 title \"%s \xe2\x80\x94"
            " Coordinate Phase Planes\" font 'Sans Bold,14'\n\n",
            ex->stem, t_days_max, ex->mission_name);

        /* Left panel: x (km, trajectory-only range) vs vx (km/h) */
        fprintf(f,
            "set xlabel 'x  (\xc3\x97 10{^3} km)'\n"
            "set ylabel 'v_x  (\xc3\x97 10{^3} km/h)'\n"
            "set title \"x  vs  x'\"\n"
            "set xrange [%g:%g]\n"
            "set yrange [%g:%g]\n",
            xtmin - tmargin, xtmax + tmargin,
            vxmin - vx_pad, vxmax + vx_pad);
        write_scaled_tics(f, 'x', xtmin - tmargin, xtmax + tmargin);
        write_scaled_tics(f, 'y', vxmin - vx_pad, vxmax + vx_pad);
        fprintf(f,
            "plot '%s' using 2:4:($1/24) "
            "with lines lw 2 lc palette notitle\n\n",
            csv_path);

        /* Right panel: y (km) vs vy (km/h) — no duplicate colorbar */
        fprintf(f,
            "unset colorbox\n"
            "set xlabel 'y  (\xc3\x97 10{^3} km)'\n"
            "set ylabel 'v_y  (\xc3\x97 10{^3} km/h)'\n"
            "set title \"y  vs  y'\"\n"
            "set xrange [%g:%g]\n"
            "set yrange [%g:%g]\n",
            ytmin - tmargin, ytmax + tmargin,
            vymin - vy_pad, vymax + vy_pad);
        write_scaled_tics(f, 'x', ytmin - tmargin, ytmax + tmargin);
        write_scaled_tics(f, 'y', vymin - vy_pad, vymax + vy_pad);
        fprintf(f,
            "plot '%s' using 3:5:($1/24) "
            "with lines lw 2 lc palette notitle\n\n",
            csv_path);

        fprintf(f,
            "unset multiplot\n"
            "unset xzeroaxis\n"
            "unset yzeroaxis\n\n");

        /* Reset before phase-space portrait */
        fprintf(f,
            "unset colorbox\n"
            "set xrange [*:*]\n"
            "set yrange [*:*]\n\n");
    }

    /* ── Figure 3: Phase-space portrait ──────────────────────────────
     *
     * Rebuild body array from stored export data, then integrate a
     * 5x5 grid of trajectories whose initial velocities are sampled
     * +-SPREAD around the actual launch velocity.  All curves are
     * embedded as gnuplot inline data blocks so the .plt is fully
     * self-contained -- no re-running of the simulator required.
     *
     * The actual mission trajectory is overlaid in magenta on top.
     * ──────────────────────────────────────────────────────────────── */

    /* Rebuild Body[] from stored export data */
    Body portrait_bodies[8];
    memset(portrait_bodies, 0, sizeof portrait_bodies);
    for (int b = 0; b < ex->n_bodies; b++) {
        portrait_bodies[b].pos.x  = ex->body_px[b];
        portrait_bodies[b].pos.y  = ex->body_py[b];
        portrait_bodies[b].mass   = ex->body_mass[b];
        portrait_bodies[b].radius = ex->body_radius[b];
    }

    /* Integration parameters */
    double x0      = ex->px[0];
    double y0      = ex->py[0];
    double vx_nom  = ex->vx[0];
    double vy_nom  = ex->vy[0];
    double spd_nom = sqrt(vx_nom * vx_nom + vy_nom * vy_nom);
    if (spd_nom < 1.0) spd_nom = 1.0;
    double t_start = ex->time[0];
    double t_end   = ex->time[ex->count - 1];
    double t_span  = t_end - t_start;
    double dt_p    = ex->dt_sim;                  /* same accuracy as original sim */
    int    max_pts = (int)(t_span / dt_p) + 1;   /* exactly cover the full duration */
    if (max_pts > 50000) max_pts = 50000;         /* guard against absurd run lengths */

    /* ── Phase-space grid: orbital radius × speed scale ─────────────
     *
     * Axis 1 (rows):    5 radial distances from the nearest body,
     *   {0.50, 0.75, 1.00, 1.25, 1.50} × r_nominal.
     *   Trajectories starting inside the body radius terminate at
     *   collision (short arcs that mark the lower phase-space boundary).
     *   Trajectories at larger r produce longer, higher arcs.
     *
     * Axis 2 (columns): 5 speed-scale factors {0.90 … 1.10} × spd_nom
     *   Lower speed → bound/shorter orbit; higher → escape trajectory.
     *
     * The (row=2, col=2) trajectory exactly reproduces the mission path.
     * Each of the 25 conditions is unique in the full 4D phase space
     * (x, y, vx, vy); their 2D spatial projections are globally distinct.
     * ──────────────────────────────────────────────────────────────── */

    /* Find nearest body — defines the reference orbital radius */
    int    b_near = 0;
    double rmin   = 1e30;
    for (int b = 0; b < ex->n_bodies; b++) {
        double ddx = x0 - portrait_bodies[b].pos.x;
        double ddy = y0 - portrait_bodies[b].pos.y;
        double r   = sqrt(ddx * ddx + ddy * ddy);
        if (r < rmin) { rmin = r; b_near = b; }
    }
    double bx     = portrait_bodies[b_near].pos.x;
    double by     = portrait_bodies[b_near].pos.y;
    double rhat_x = (x0 - bx) / rmin;   /* radial unit vector (body → spacecraft) */
    double rhat_y = (y0 - by) / rmin;

    /* Grid parameters */
#define PG_N    5
#define RAD_LO  0.50   /* innermost radial-scale factor */
#define RAD_HI  1.50   /* outermost radial-scale factor */
#define SPD_LO  0.90   /* minimum speed-scale factor   */
#define SPD_HI  1.10   /* maximum speed-scale factor   */
    double drad = (PG_N > 1) ? ((RAD_HI - RAD_LO) / (PG_N - 1)) : 0.0;
    double dspd = (PG_N > 1) ? ((SPD_HI - SPD_LO) / (PG_N - 1)) : 0.0;

    /* Viewport: nominal trajectory bounds expanded for outer starting positions */
    double pos_expand = (RAD_HI - 1.0) * rmin;
    double px_lo = xmin - margin - pos_expand;
    double px_hi = xmax + margin + pos_expand;
    double py_lo = ymin - margin - pos_expand;
    double py_hi = ymax + margin + pos_expand;

    /* Emit each integral curve as a comma-separated gnuplot inline block */
    int ci = 0;
    for (int ir = 0; ir < PG_N; ir++) {          /* orbital radius row */
        double r_i = rmin * (RAD_LO + ir * drad);
        double xi  = bx + rhat_x * r_i;
        double yi  = by + rhat_y * r_i;

        for (int iv = 0; iv < PG_N; iv++) {       /* speed scale column */
            double spd_scale = SPD_LO + iv * dspd;
            double vx_i = vx_nom * spd_scale;
            double vy_i = vy_nom * spd_scale;

            Spacecraft mini;
            mini.pos.x             = xi;
            mini.pos.y             = yi;
            mini.vel.x             = vx_i;
            mini.vel.y             = vy_i;
            mini.acc               = resultant_acc(portrait_bodies, ex->n_bodies, mini.pos);
            mini.distance_traveled = 0.0;
            mini.time              = t_start;

            /* Phase-portrait inline data: raw km positions */
            fprintf(f, "$C%d << EOD\n", ci);
            fprintf(f, "%.3f,%.3f\n", mini.pos.x, mini.pos.y);
            for (int sp = 0; sp < max_pts; sp++) {
                step_verlet(&mini, portrait_bodies, ex->n_bodies, dt_p);
                fprintf(f, "%.3f,%.3f\n", mini.pos.x, mini.pos.y);
                if (check_collision(&mini, portrait_bodies, ex->n_bodies) >= 0) break;
            }
            fprintf(f, "EOD\n\n");
            ci++;
        }
    }
    /* Save grid constants before #undef so they are available in the plot loop */
    const int    pgn      = PG_N;
    const double rad_lo_v = RAD_LO;
    const double drad_v   = drad;
#undef PG_N
#undef RAD_LO
#undef RAD_HI
#undef SPD_LO
#undef SPD_HI

    /* Dynamic canvas for phase-space portrait */
    {
        double port_ar = (px_hi - px_lo) / fmax(py_hi - py_lo, 1.0);
        int c4w = (int)(900.0 * fmin(fmax(port_ar, 0.4), 2.7));
        int c4h = 900;
        if (c4w < 900) { c4h = (int)(900.0 / fmax(port_ar, 0.1)); c4w = 900; }
        if (c4w > 2400) c4w = 2400;
        if (c4h > 1200) c4h = 1200;
        fprintf(f, "set terminal pngcairo size %d,%d enhanced font 'Sans,11'\n",
                c4w, c4h);
    }

    /* Portrait body circles */
    double portrait_span = (px_hi - px_lo);
    double portrait_min_r = portrait_span * 0.002;
    fprintf(f,
        "set output 'integral_%s.png'\n"
        "set title \"%s \xe2\x80\x94 Phase-Space Portrait"
        "  (5 orbital radii \xc3\x97 5 speed levels)\""
        " font 'Sans Bold,14'\n"
        "set xlabel 'X  (\xc3\x97 10{^3} km)'\n"
        "set ylabel 'Y  (\xc3\x97 10{^3} km)'\n"
        "set size ratio -1\n"
        "set grid lc rgb '#333333' lw 0.4\n"
        "set xrange [%g:%g]\n"
        "set yrange [%g:%g]\n"
        "unset colorbox\n",
        ex->stem,
        ex->mission_name,
        px_lo, px_hi, py_lo, py_hi);

    write_scaled_tics(f, 'x', px_lo, px_hi);
    write_scaled_tics(f, 'y', py_lo, py_hi);
    fprintf(f, "\n");

    /* Body circles: show motion in rotating frame at start, middle, and end */
    for (int b = 0; b < ex->n_bodies; b++) {
        double r = ex->body_radius[b];
        if (r < portrait_min_r) r = portrait_min_r;
        
        int start_idx  = 0;
        int mid_idx    = ex->count / 2;
        int end_idx    = ex->count - 1;
        if (end_idx < 0) end_idx = 0;
        
        /* Start: full opacity (use object ID offset by 100) */
        fprintf(f, "set object %d circle at %g,%g size %g"
                   " fillstyle solid 1.0 fillcolor rgb '%s' front\n",
                100 + b + 1, ex->body_px_hist[b][start_idx], ex->body_py_hist[b][start_idx], r,
                BODY_COLORS[b < 8 ? b : 7]);
        
        /* Middle: medium opacity */
        if (mid_idx > start_idx) {
            fprintf(f, "set object %d circle at %g,%g size %g"
                       " fillstyle solid 0.5 fillcolor rgb '%s' front\n",
                    100 + ex->n_bodies + b + 1, ex->body_px_hist[b][mid_idx], ex->body_py_hist[b][mid_idx], r,
                    BODY_COLORS[b < 8 ? b : 7]);
        }
        
        /* End: low opacity */
        if (end_idx > start_idx) {
            fprintf(f, "set object %d circle at %g,%g size %g"
                       " fillstyle solid 0.2 fillcolor rgb '%s' front\n",
                    100 + 2*ex->n_bodies + b + 1, ex->body_px_hist[b][end_idx], ex->body_py_hist[b][end_idx], r,
                    BODY_COLORS[b < 8 ? b : 7]);
        }
        
        /* Label at end position */
        int label_idx = end_idx;
        fprintf(f,
            "set label %d '%s' at %g,%g "
            "tc rgb '%s' front offset character 0.6,0.6\n",
            b + 1, ex->body_names[b],
            ex->body_px_hist[b][label_idx], ex->body_py_hist[b][label_idx],
            BODY_COLORS[b < 8 ? b : 7]);
    }

    /* Integral curves: colour by orbital-radius factor, width by speed factor */
    static const char * const port_rc[5] = {
        "#3344cc", "#3399ff", "#999999", "#ff8833", "#cc3333"
    };
    static const double port_lw[5] = { 0.7, 1.0, 1.5, 1.0, 0.7 };

    fprintf(f, "plot ");
    {
        int ci2 = 0;
        for (int ir2 = 0; ir2 < pgn; ir2++) {
            for (int iv2 = 0; iv2 < pgn; iv2++) {
                fprintf(f,
                    "$C%d using 1:2 with lines lw %g lc rgb '%s' notitle,\\\n     ",
                    ci2, port_lw[iv2], port_rc[ir2]);
                ci2++;
            }
        }
    }
    /* Orbital-radius legend entries (1/0 = invisible point, legend entry only) */
    for (int ir2 = 0; ir2 < pgn; ir2++) {
        double rf = rad_lo_v + ir2 * drad_v;
        fprintf(f, "1/0 with lines lw 2 lc rgb '%s' title '%.2f\xc3\x97r_0',\\\n     ",
                port_rc[ir2], rf);
    }
    /* Body trajectories (dashed) behind the mission path */
    for (int b = 0; b < ex->n_bodies; b++) {
        int bx_col = 7 + ex->n_bodies + 2 * b;
        int by_col = bx_col + 1;
        fprintf(f,
            "'%s' using %d:%d with lines "
            "lw 1.5 lc rgb '%s' dt 2 notitle,\\\n     ",
            csv_path, bx_col, by_col,
            BODY_COLORS[b < 8 ? b : 7]);
    }
    fprintf(f,
        "'%s' using 2:3 with lines lw 3 lc rgb '#ff00ff' "
        "title 'Mission',\\\n"
        "     '%s' using ($0==1?$2:1/0):($0==1?$3:1/0) "
        "with points pt 7 ps 2.5 lc rgb '#00ff00' title 'Start',\\\n"
        "     '%s' using ($0==%d?$2:1/0):($0==%d?$3:1/0) "
        "with points pt 2 ps 2.5 lc rgb '#ff4444' title 'End'\n",
        csv_path, csv_path, csv_path, ex->count, ex->count);

    for (int b = 0; b < ex->n_bodies; b++) {
        /* Clean up trajectory bodies (ID 0-7, 100-107, 108-115 for start/mid/end) */
        fprintf(f, "unset object %d\n", b + 1);
        fprintf(f, "unset object %d\n", 100 + b + 1);
        fprintf(f, "unset object %d\n", 100 + ex->n_bodies + b + 1);
        fprintf(f, "unset object %d\n", 100 + 2*ex->n_bodies + b + 1);
        fprintf(f, "unset label %d\n", b + 1);
    }

    fclose(f);
    return 0;
}


static void write_animation_c(const Exporter *ex, const char *csv_path)
{
    /* Extract directory from csv_path by finding the last '/' */
    char out_dir[512] = "";
    const char *last_slash = strrchr(csv_path, '/');
    if (last_slash) {
        size_t dir_len = (size_t)(last_slash - csv_path);
        if (dir_len < sizeof out_dir) {
            strncpy(out_dir, csv_path, dir_len);
            out_dir[dir_len] = '\0';
        }
    }

    char plt_path[512];
    char sh_path [512];
    char c_path[512];

    if (out_dir[0]) {
        snprintf(plt_path, sizeof plt_path, "%s/%s_movie.plt",  out_dir, ex->stem);
        snprintf(sh_path,  sizeof sh_path,  "%s/%s_animate.sh", out_dir, ex->stem);
        snprintf(c_path, sizeof c_path, "%s/%s_animate.c", out_dir, ex->stem);
    } else {
        snprintf(plt_path, sizeof plt_path, "%s_movie.plt",  ex->stem);
        snprintf(sh_path,  sizeof sh_path,  "%s_animate.sh", ex->stem);
        snprintf(c_path, sizeof c_path, "%s_animate.c", ex->stem);
    }

    FILE *f = fopen(c_path, "w");
    if (!f) return;

    /* ── viewport ── */
    double xmin = ex->px[0], xmax = ex->px[0];
    double ymin = ex->py[0], ymax = ex->py[0];
    for (int i = 1; i < ex->count; i++) {
        if (ex->px[i] < xmin) xmin = ex->px[i];
        if (ex->px[i] > xmax) xmax = ex->px[i];
        if (ex->py[i] < ymin) ymin = ex->py[i];
        if (ex->py[i] > ymax) ymax = ex->py[i];
    }
    for (int b = 0; b < ex->n_bodies; b++) {
        for (int i = 0; i < ex->count; i++) {
            if (ex->body_px_hist[b][i] < xmin) xmin = ex->body_px_hist[b][i];
            if (ex->body_px_hist[b][i] > xmax) xmax = ex->body_px_hist[b][i];
            if (ex->body_py_hist[b][i] < ymin) ymin = ex->body_py_hist[b][i];
            if (ex->body_py_hist[b][i] > ymax) ymax = ex->body_py_hist[b][i];
        }
    }
    double margin = (xmax - xmin) * 0.08 + 5000.0;

    /* ── parse body hex colors → RGB at generation time ── */
    int bcr[8], bcg[8], bcb[8];
    for (int b = 0; b < ex->n_bodies && b < 8; b++) {
        unsigned int rv = 0, gv = 0, bv = 0;
        sscanf(BODY_COLORS[b], "#%02x%02x%02x", &rv, &gv, &bv);
        bcr[b] = (int)rv;  bcg[b] = (int)gv;  bcb[b] = (int)bv;
    }

    /* ── header + includes ── */
    fprintf(f,
        "/*\n"
        " * %s_animate.c — auto-generated trajectory animation\n"
        " * Mission : %s\n"
        " *\n"
        " * Build:  cc -O2 -o %s_animate %s_animate.c -lm\n"
        " * Run:    ./%s_animate\n"
        " * Needs:  ffmpeg in PATH, %s in the same directory\n"
        " */\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "#include <math.h>\n\n",
        ex->stem, ex->mission_name,
        ex->stem, ex->stem,
        ex->stem, csv_path);

    /* ── sizing constants ── */
    fputs(
        "#define FRAME_W    1200\n"
        "#define FRAME_H     900\n"
        "#define FPS          30\n"
        "#define MAX_FRAMES  600\n"
        "#define MAX_ROWS  500000\n\n",
        f);

    /* ── viewport (embedded from simulation) ── */
    fprintf(f,
        "/* viewport in km, captured at export time */\n"
        "static const double XMIN = %.17g;\n"
        "static const double XMAX = %.17g;\n"
        "static const double YMIN = %.17g;\n"
        "static const double YMAX = %.17g;\n\n",
        xmin - margin, xmax + margin,
        ymin - margin, ymax + margin);

    /* ── body data (embedded from simulation) ── */
    fprintf(f, "#define N_BODIES %d\n", ex->n_bodies);

    /* Initial body positions — fallback if CSV body columns absent */
    fprintf(f, "static const double BX0[N_BODIES] = {");
    for (int b = 0; b < ex->n_bodies; b++)
        fprintf(f, "%s%.17g", b ? "," : "", ex->body_px[b]);
    fputs("}; /* initial x positions (km) */\n", f);

    fprintf(f, "static const double BY0[N_BODIES] = {");
    for (int b = 0; b < ex->n_bodies; b++)
        fprintf(f, "%s%.17g", b ? "," : "", ex->body_py[b]);
    fputs("}; /* initial y positions (km) */\n", f);

    fprintf(f, "static const unsigned char BC[N_BODIES][3] = {");
    for (int b = 0; b < ex->n_bodies; b++)
        fprintf(f, "%s{%d,%d,%d}", b ? "," : "", bcr[b], bcg[b], bcb[b]);
    fputs("}; /* RGB colours */\n", f);

    fprintf(f, "static const double BR[N_BODIES] = {");
    for (int b = 0; b < ex->n_bodies; b++)
        fprintf(f, "%s%.17g", b ? "," : "", ex->body_radius[b]);
    fputs("}; /* physical radii (km) */\n\n", f);

    /* ── paths (embedded) ── */
    /* Extract just the filename from csv_path for relative path within mission directory */
    const char *csv_filename = strrchr(csv_path, '/');
    csv_filename = csv_filename ? csv_filename + 1 : csv_path;  /* skip directory, use just filename */
    fprintf(f,
        "static const char CSV_IN[]  = \"%s\";\n"
        "static const char MP4_OUT[] = \"%s.mp4\";\n\n",
        csv_filename, ex->stem);

    /* ── static buffers ── */
    fputs(
        "/* frame buffer: RGB24, row-major */\n"
        "static unsigned char fb[FRAME_H][FRAME_W][3];\n\n"
        "/* trajectory data loaded from CSV */\n"
        "static int    nrows;\n"
        "static double row_t[MAX_ROWS], row_x[MAX_ROWS], row_y[MAX_ROWS];\n"
        "static double row_vx[MAX_ROWS], row_vy[MAX_ROWS], row_spd[MAX_ROWS];\n"
        "static double row_bx[MAX_ROWS][N_BODIES]; /* body x per row */\n"
        "static double row_by[MAX_ROWS][N_BODIES]; /* body y per row */\n"
        "static double spd_max;\n\n",
        f);

    /* ── w2s ── */
    fputs(
        "static void w2s(double wx, double wy, int *sx, int *sy)\n"
        "{\n"
        "    *sx = (int)((wx - XMIN) / (XMAX - XMIN) * FRAME_W);\n"
        "    *sy = FRAME_H - 1 - (int)((wy - YMIN) / (YMAX - YMIN) * FRAME_H);\n"
        "}\n\n",
        f);

    /* ── fb_set (clipped pixel write) ── */
    fputs(
        "static void fb_set(int x, int y,\n"
        "                   unsigned char r, unsigned char g, unsigned char b)\n"
        "{\n"
        "    if ((unsigned)x >= FRAME_W || (unsigned)y >= FRAME_H) return;\n"
        "    fb[y][x][0] = r;  fb[y][x][1] = g;  fb[y][x][2] = b;\n"
        "}\n\n",
        f);

    /* ── pal (speed → blue→cyan→yellow→red palette) ── */
    fputs(
        "static void pal(double spd,\n"
        "                unsigned char *r, unsigned char *g, unsigned char *b)\n"
        "{\n"
        "    double t = (spd_max > 0.0) ? spd / spd_max : 0.0;\n"
        "    if (t < 0.0) t = 0.0;\n"
        "    if (t > 1.0) t = 1.0;\n"
        "    double R, G, B;\n"
        "    if (t < 0.33) {\n"
        "        double u = t / 0.33;\n"
        "        R = 0.0; G = u;       B = 1.0;\n"
        "    } else if (t < 0.66) {\n"
        "        double u = (t - 0.33) / 0.33;\n"
        "        R = u;   G = 1.0;     B = 1.0 - u;\n"
        "    } else {\n"
        "        double u = (t - 0.66) / 0.34;\n"
        "        R = 1.0; G = 1.0 - u; B = 0.0;\n"
        "    }\n"
        "    *r = (unsigned char)(R * 255);\n"
        "    *g = (unsigned char)(G * 255);\n"
        "    *b = (unsigned char)(B * 255);\n"
        "}\n\n",
        f);

    /* ── draw_ln (Bresenham line) ── */
    fputs(
        "static void draw_ln(int x0, int y0, int x1, int y1,\n"
        "                    unsigned char r, unsigned char g, unsigned char b)\n"
        "{\n"
        "    int dx = abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;\n"
        "    int dy = -abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;\n"
        "    int err = dx + dy, e2;\n"
        "    for (;;) {\n"
        "        fb_set(x0, y0, r, g, b);\n"
        "        if (x0 == x1 && y0 == y1) break;\n"
        "        e2 = 2 * err;\n"
        "        if (e2 >= dy) { if (x0 == x1) break; err += dy; x0 += sx; }\n"
        "        if (e2 <= dx) { if (y0 == y1) break; err += dx; y0 += sy; }\n"
        "    }\n"
        "}\n\n",
        f);

    /* ── draw_disc (filled circle) ── */
    fputs(
        "static void draw_disc(int cx, int cy, int rad,\n"
        "                      unsigned char r, unsigned char g, unsigned char b)\n"
        "{\n"
        "    int qi, qj;\n"
        "    for (qi = -rad; qi <= rad; qi++)\n"
        "        for (qj = -rad; qj <= rad; qj++)\n"
        "            if (qi * qi + qj * qj <= rad * rad)\n"
        "                fb_set(cx + qj, cy + qi, r, g, b);\n"
        "}\n\n",
        f);

    /* ── draw_tri (filled triangle, edge-function rasteriser) ── */
    fputs(
        "static void draw_tri(int ax, int ay, int bx, int by, int cx, int cy,\n"
        "                     unsigned char r, unsigned char g, unsigned char bl)\n"
        "{\n"
        "    int x0 = ax, x1 = ax, y0 = ay, y1 = ay;\n"
        "    if (bx < x0) x0 = bx;\n"
        "    if (bx > x1) x1 = bx;\n"
        "    if (cx < x0) x0 = cx;\n"
        "    if (cx > x1) x1 = cx;\n"
        "    if (by < y0) y0 = by;\n"
        "    if (by > y1) y1 = by;\n"
        "    if (cy < y0) y0 = cy;\n"
        "    if (cy > y1) y1 = cy;\n"
        "    int qi, qj, d1, d2, d3, hn, hp;\n"
        "    for (qi = y0; qi <= y1; qi++) {\n"
        "        for (qj = x0; qj <= x1; qj++) {\n"
        "            d1 = (bx-ax)*(qi-ay) - (by-ay)*(qj-ax);\n"
        "            d2 = (cx-bx)*(qi-by) - (cy-by)*(qj-bx);\n"
        "            d3 = (ax-cx)*(qi-cy) - (ay-cy)*(qj-cx);\n"
        "            hn = (d1 < 0) || (d2 < 0) || (d3 < 0);\n"
        "            hp = (d1 > 0) || (d2 > 0) || (d3 > 0);\n"
        "            if (!(hn && hp)) fb_set(qj, qi, r, g, bl);\n"
        "        }\n"
        "    }\n"
        "}\n\n",
        f);

    /* ── draw_spacecraft (yellow triangle, tip points in velocity direction) ── */
    fputs(
        "static void draw_spacecraft(int si)\n"
        "{\n"
        "    int sx, sy;\n"
        "    w2s(row_x[si], row_y[si], &sx, &sy);\n"
        "    /* flip vy: screen y increases downward, world y increases upward */\n"
        "    double dvx = row_vx[si], dvy = -row_vy[si];\n"
        "    double dl  = sqrt(dvx * dvx + dvy * dvy);\n"
        "    if (dl < 1e-10) { dvx = 0.0; dvy = -1.0; }\n"
        "    else             { dvx /= dl;  dvy /= dl; }\n"
        "    double pex = -dvy, pey = dvx;  /* perpendicular */\n"
        "    double sz  = FRAME_W * 0.014;\n"
        "    int tx = (int)(sx + dvx * sz * 1.4);\n"
        "    int ty = (int)(sy + dvy * sz * 1.4);\n"
        "    int lx = (int)(sx - dvx * sz * 0.7 + pex * sz * 0.65);\n"
        "    int ly = (int)(sy - dvy * sz * 0.7 + pey * sz * 0.65);\n"
        "    int rx = (int)(sx - dvx * sz * 0.7 - pex * sz * 0.65);\n"
        "    int ry = (int)(sy - dvy * sz * 0.7 - pey * sz * 0.65);\n"
        "    draw_tri(tx, ty, lx, ly, rx, ry, 255, 220, 0);\n"
        "}\n\n",
        f);

    /* ── render_frame ── */
    fputs(
        "static void render_frame(int si)\n"
        "{\n"
        "    int i;\n"
        "    unsigned char r, g, b;\n"
        "    memset(fb, 0, sizeof fb);\n\n"
        "    /* growing speed-coloured trail */\n"
        "    for (i = 0; i < si && i + 1 < nrows; i++) {\n"
        "        int ax, ay, ex2, ey;\n"
        "        w2s(row_x[i],     row_y[i],     &ax,  &ay);\n"
        "        w2s(row_x[i + 1], row_y[i + 1], &ex2, &ey);\n"
        "        pal(row_spd[i], &r, &g, &b);\n"
        "        draw_ln(ax, ay, ex2, ey, r, g, b);\n"
        "    }\n\n"
        "    /* body markers — position advances each frame */\n"
        "    for (i = 0; i < N_BODIES; i++) {\n"
        "        int bsx, bsy;\n"
        "        w2s(row_bx[si][i], row_by[si][i], &bsx, &bsy);\n"
        "        int brad = (int)(BR[i] / (XMAX - XMIN) * FRAME_W);\n"
        "        if (brad < 3) brad = 3;\n"
        "        draw_disc(bsx, bsy, brad, BC[i][0], BC[i][1], BC[i][2]);\n"
        "    }\n\n"
        "    /* spacecraft */\n"
        "    if (si < nrows) draw_spacecraft(si);\n"
        "}\n\n",
        f);

    /* ── load_csv ── */
    /* Emit the number of dist columns so we can skip them to reach body-xy cols */
    fprintf(f, "#define N_DIST_COLS %d  /* == N_BODIES */\n\n", ex->n_bodies);
    fputs(
        "static int load_csv(void)\n"
        "{\n"
        "    FILE *fp = fopen(CSV_IN, \"r\");\n"
        "    if (!fp) { fprintf(stderr, \"Cannot open %s\\n\", CSV_IN); return -1; }\n"
        "    char buf[4096];\n"
        "    /* Skip two header lines */\n"
        "    if (!fgets(buf, sizeof buf, fp)) { fclose(fp); return 0; }\n"
        "    if (!fgets(buf, sizeof buf, fp)) { fclose(fp); return 0; }\n"
        "    nrows = 0;  spd_max = 0.0;\n"
        "    /* Initialise body positions from compile-time fallback */\n"
        "    int i2;\n"
        "    for (i2 = 0; i2 < N_BODIES; i2++) {\n"
        "        row_bx[0][i2] = BX0[i2];\n"
        "        row_by[0][i2] = BY0[i2];\n"
        "    }\n"
        "    while (fgets(buf, sizeof buf, fp) && nrows < MAX_ROWS) {\n"
        "        if (buf[0] == '#') continue;\n"
        "        double t, x, y, vx2, vy2, spd;\n"
        "        /* Parse the first 6 fixed columns */\n"
        "        char *p = buf;\n"
        "        int ok = sscanf(p, \"%lf,%lf,%lf,%lf,%lf,%lf\",\n"
        "                        &t, &x, &y, &vx2, &vy2, &spd) == 6;\n"
        "        if (!ok) continue;\n"
        "        row_t[nrows] = t;  row_x[nrows] = x;   row_y[nrows] = y;\n"
        "        row_vx[nrows] = vx2; row_vy[nrows] = vy2; row_spd[nrows] = spd;\n"
        "        if (spd > spd_max) spd_max = spd;\n"
        "        /* Skip past 6 + N_DIST_COLS comma-fields to reach body-xy */\n"
        "        int skip = 6 + N_DIST_COLS, ci;\n"
        "        for (ci = 0; ci < skip && p; ci++) { p = strchr(p, ','); if (p) p++; }\n"
        "        int bj;\n"
        "        for (bj = 0; bj < N_BODIES && p; bj++) {\n"
        "            double bx2 = BX0[bj], by2 = BY0[bj];\n"
        "            char *comma;\n"
        "            if (sscanf(p, \"%lf\", &bx2) == 1)\n"
        "                row_bx[nrows][bj] = bx2;\n"
        "            else\n"
        "                row_bx[nrows][bj] = nrows > 0 ? row_bx[nrows-1][bj] : BX0[bj];\n"
        "            /* advance past bx field */\n"
        "            comma = strchr(p, ','); if (!comma) break; p = comma + 1;\n"
        "            if (sscanf(p, \"%lf\", &by2) == 1)\n"
        "                row_by[nrows][bj] = by2;\n"
        "            else\n"
        "                row_by[nrows][bj] = nrows > 0 ? row_by[nrows-1][bj] : BY0[bj];\n"
        "            /* advance past by field */\n"
        "            comma = strchr(p, ','); p = comma ? comma + 1 : NULL;\n"
        "        }\n"
        "        nrows++;\n"
        "    }\n"
        "    fclose(fp);\n"
        "    return nrows;\n"
        "}\n\n",
        f);

    /* ── main ── */
    fputs(
        "int main(void)\n"
        "{\n"
        "    if (load_csv() <= 0) return 1;\n\n"
        "    int step = (nrows + MAX_FRAMES - 1) / MAX_FRAMES;\n"
        "    if (step < 1) step = 1;\n"
        "    int nf = (nrows + step - 1) / step;\n\n"
        "    char cmd[512];\n"
        "    snprintf(cmd, sizeof cmd,\n"
        "        \"ffmpeg -y -f rawvideo -vcodec rawvideo\"\n"
        "        \" -s %dx%d -pix_fmt rgb24 -r %d -i pipe:0\"\n"
        "        \" -c:v libx264 -pix_fmt yuv420p -crf 22 %s\",\n"
        "        FRAME_W, FRAME_H, FPS, MP4_OUT);\n\n"
        "    FILE *pipe2 = popen(cmd, \"w\");\n"
        "    if (!pipe2) { fprintf(stderr, \"Cannot open ffmpeg pipe\\n\"); return 1; }\n\n"
        "    fprintf(stderr, \"Rendering %d frames  (%d data rows, step %d)...\\n\",\n"
        "            nf, nrows, step);\n\n"
        "    int fi;\n"
        "    for (fi = 0; fi < nf; fi++) {\n"
        "        int si = fi * step;\n"
        "        if (si >= nrows) si = nrows - 1;\n"
        "        render_frame(si);\n"
        "        fwrite(fb, 1, sizeof fb, pipe2);\n"
        "        if ((fi + 1) % 30 == 0 || fi + 1 == nf)\n"
        "            fprintf(stderr, \"  frame %d/%d\\r\", fi + 1, nf);\n"
        "    }\n"
        "    fprintf(stderr, \"\\n\");\n\n"
        "    int rc = pclose(pipe2);\n"
        "    if (rc == 0)\n"
        "        fprintf(stderr, \"Done: %s\\n\", MP4_OUT);\n"
        "    else\n"
        "        fprintf(stderr, \"ffmpeg exited with code %d\\n\", rc);\n"
        "    return (rc == 0) ? 0 : 1;\n"
        "}\n",
        f);

    fclose(f);
}

/* ── public write ────────────────────────────────────────────── */

/* ── render: run gnuplot and/or compile+execute the C animator ─ */

int exporter_render(const Exporter *ex, int flags)
{
    if (ex->count == 0 || flags == 0) return 0;

    /* Validate stem: only alphanumeric, '_', '-', '.' allowed.
     * This keeps the system() commands safe from shell injection. */
    for (const char *p = ex->stem; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (!isalnum(c) && c != '_' && c != '-' && c != '.') {
            fprintf(stderr,
                "export: unsafe stem '%s' — skipping media render.\n",
                ex->stem);
            return 0;
        }
    }

    char cmd[EXPORT_STEM_MAX * 4 + 64];
    int done = 0;

    if (flags & EXPORT_IMAGES) {
        snprintf(cmd, sizeof cmd, "gnuplot '%s.plt'", ex->stem);
        if (system(cmd) == 0) done |= EXPORT_IMAGES;
    }

    if (flags & EXPORT_VIDEO) {
        /* Compile the generated C animator, then run it. */
        snprintf(cmd, sizeof cmd,
            "cc -O2 -o '%s_animate' '%s_animate.c' -lm && './%s_animate'",
            ex->stem, ex->stem, ex->stem);
        if (system(cmd) == 0) done |= EXPORT_VIDEO;
    }

    return done;
}

/* ── public write ────────────────────────────────────────────── */

int exporter_write(const Exporter *ex)
{
    if (ex->count == 0) return -1;

    /* Construct output directory path: missions/output/<mission_name>/ */
    char out_dir[512];
    snprintf(out_dir, sizeof out_dir, "missions/output/%s", ex->mission_name);

    /* Create output directory if it doesn't exist */
    char mkdir_cmd[1024];
    snprintf(mkdir_cmd, sizeof mkdir_cmd, "mkdir -p %s", out_dir);
    if (system(mkdir_cmd) < 0) return -1;  /* Suppress unused-result warning */

    /* Construct full file paths */
    char csv_path[768];
    char plt_path[768];
    snprintf(csv_path, sizeof csv_path, "%s/%s.csv", out_dir, ex->stem);
    snprintf(plt_path, sizeof plt_path, "%s/%s.plt", out_dir, ex->stem);

    if (write_csv(ex, csv_path) < 0)  return -1;
    if (write_gnuplot(ex, csv_path, plt_path) < 0) return -1;
    write_animation_c(ex, csv_path);
    return 0;
}

/* ── free ────────────────────────────────────────────────────── */

void exporter_free(Exporter *ex)
{
    free(ex->time);
    free(ex->px); free(ex->py);
    free(ex->vx); free(ex->vy);
    free(ex->speed);
    for (int i = 0; i < 8; i++) {
        free(ex->dist[i]);
        free(ex->body_px_hist[i]);
        free(ex->body_py_hist[i]);
    }
    memset(ex, 0, sizeof *ex);
}
