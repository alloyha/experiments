/*
 * main.c — Spacecraft Trajectory Simulator  (interactive + batch)
 *
 *  ./simulator          →  full ncurses interactive mode
 *  ./simulator file.txt →  batch command processing
 */

#include "physics.h"
#include "bodies.h"
#include "missions.h"
#include "export.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <ncurses.h>

/* ═══════════════════════════════════════════════════════════════
 * Timing / frame-rate
 * ═══════════════════════════════════════════════════════════════ */

#define TARGET_FPS   30
#define FRAME_US     (1000000L / TARGET_FPS)
#define MIN_STEPS    1
#define MAX_STEPS    50000

static void sleep_us(long us)
{
    struct timespec ts = { us / 1000000L, (us % 1000000L) * 1000L };
    nanosleep(&ts, NULL);
}

/* ═══════════════════════════════════════════════════════════════
 * Trail (circular buffer)
 * ═══════════════════════════════════════════════════════════════ */

#define TRAIL_MAX 4096

typedef struct {
    Vec2 pts[TRAIL_MAX];
    int  head;
    int  count;
} Trail;

static void trail_push(Trail *t, Vec2 p)
{
    t->pts[t->head] = p;
    t->head = (t->head + 1) % TRAIL_MAX;
    if (t->count < TRAIL_MAX) t->count++;
}

static Vec2 trail_get(const Trail *t, int age)
{
    return t->pts[(t->head - 1 - age + TRAIL_MAX) % TRAIL_MAX];
}

/* ═══════════════════════════════════════════════════════════════
 * ncurses colour pairs
 * ═══════════════════════════════════════════════════════════════ */

enum {
    CP_EARTH = 1, CP_MOON, CP_SUN, CP_BODY4,
    CP_CRAFT, CP_TRAIL_HOT, CP_TRAIL_MID, CP_TRAIL_COLD,
    CP_L1, CP_BORDER, CP_LABEL, CP_VALUE,
    CP_TITLE, CP_ALERT, CP_OK,
};

static void init_colors(void)
{
    start_color();
    use_default_colors();
    init_pair(CP_EARTH,      COLOR_CYAN,    -1);
    init_pair(CP_MOON,       COLOR_WHITE,   -1);
    init_pair(CP_SUN,        COLOR_YELLOW,  -1);
    init_pair(CP_BODY4,      COLOR_GREEN,   -1);
    init_pair(CP_CRAFT,      COLOR_YELLOW,  -1);
    init_pair(CP_TRAIL_HOT,  COLOR_GREEN,   -1);
    init_pair(CP_TRAIL_MID,  COLOR_GREEN,   -1);
    init_pair(CP_TRAIL_COLD, COLOR_GREEN,   -1);
    init_pair(CP_L1,         COLOR_MAGENTA, -1);
    init_pair(CP_BORDER,     COLOR_CYAN,    -1);
    init_pair(CP_LABEL,      COLOR_CYAN,    -1);
    init_pair(CP_VALUE,      COLOR_WHITE,   -1);
    init_pair(CP_TITLE,      COLOR_YELLOW,  -1);
    init_pair(CP_ALERT,      COLOR_RED,     -1);
    init_pair(CP_OK,         COLOR_GREEN,   -1);
}

/* ═══════════════════════════════════════════════════════════════
 * Viewport
 * ═══════════════════════════════════════════════════════════════ */

typedef struct { double x_min, x_max, y_min, y_max; } Viewport;

static Viewport make_viewport(double cx, double cy, double span, int cols, int rows)
{
    double half_x = span / 2.0 * 1.1;
    double km_per_col = (2.0 * half_x) / (double)cols;
    double km_per_row = km_per_col * 2.0;
    double half_y     = km_per_row * rows / 2.0;
    return (Viewport){ cx - half_x, cx + half_x, cy - half_y, cy + half_y };
}

static int wc(double wx, int cols, const Viewport *vp)
{
    int c = (int)((wx - vp->x_min) / (vp->x_max - vp->x_min) * (cols-1) + 0.5);
    return c < 0 ? 0 : c >= cols ? cols-1 : c;
}
static int wr(double wy, int rows, const Viewport *vp)
{
    int r = (int)((vp->y_max - wy) / (vp->y_max - vp->y_min) * (rows-1) + 0.5);
    return r < 0 ? 0 : r >= rows ? rows-1 : r;
}

/* ═══════════════════════════════════════════════════════════════
 * Drawing helpers
 * ═══════════════════════════════════════════════════════════════ */

#define WROW_OK(w, r) ((r) >= 1 && (r) < getmaxy(w) - 1)

static void draw_box(WINDOW *w, const char *title)
{
    wattron(w, COLOR_PAIR(CP_BORDER));
    box(w, 0, 0);
    wattroff(w, COLOR_PAIR(CP_BORDER));
    int x = (getmaxx(w) - (int)strlen(title) - 2) / 2;
    if (x < 1) x = 1;
    wattron(w, COLOR_PAIR(CP_TITLE) | A_BOLD);
    mvwprintw(w, 0, x, " %s ", title);
    wattroff(w, COLOR_PAIR(CP_TITLE) | A_BOLD);
}

/* ─── Map window ─────────────────────────────────────────────── */

static void draw_map(WINDOW           *w,
                     const Scenario   *scenario,
                     const Body       *bodies, int n_bodies,
                     const Spacecraft *sc,
                     const Trail      *trail)
{
    int rows = getmaxy(w), cols = getmaxx(w);
    int ir = rows - 2, ic = cols - 2;

    wclear(w);
    draw_box(w, scenario->name);

    /* Viewport follows bodies[0] (reference body) */
    double ref_x = (n_bodies > 0) ? bodies[0].pos.x : scenario->view_cx;
    double ref_y = (n_bodies > 0) ? bodies[0].pos.y : scenario->view_cy;
    Viewport vp = make_viewport(ref_x, ref_y, scenario->view_span, ic, ir);

    /* Trail — points stored relative to bodies[0]; restore absolute coords for rendering */
    for (int age = trail->count - 1; age >= 0; age--) {
        Vec2 p = (n_bodies > 0)
                 ? v2_add(bodies[0].pos, trail_get(trail, age))
                 : trail_get(trail, age);
        int  r = wr(p.y, ir, &vp) + 1;
        int  c = wc(p.x, ic, &vp) + 1;
        double frac = (double)age / (trail->count > 1 ? trail->count - 1 : 1);
        int pair; attr_t attr;
        if      (frac < 0.25) { pair = CP_TRAIL_HOT;  attr = A_BOLD; }
        else if (frac < 0.60) { pair = CP_TRAIL_MID;  attr = 0;      }
        else                  { pair = CP_TRAIL_COLD;  attr = A_DIM;  }
        if (WROW_OK(w, r)) {
            wattron(w, COLOR_PAIR(pair) | attr);
            mvwaddch(w, r, c, '.');
            wattroff(w, COLOR_PAIR(pair) | attr);
        }
    }

    /* L1 Lagrange point */
    if (n_bodies >= 2) {
        Vec2 l1 = lagrange_l1(&bodies[0], &bodies[1]);
        int  r  = wr(l1.y, ir, &vp) + 1;
        int  c  = wc(l1.x, ic, &vp) + 1;
        if (WROW_OK(w, r)) {
            wattron(w, COLOR_PAIR(CP_L1) | A_BOLD);
            mvwaddch(w, r, c, 'x');
            wattroff(w, COLOR_PAIR(CP_L1) | A_BOLD);
        }
    }

    /* Celestial bodies */
    for (int i = 0; i < n_bodies; i++) {
        int r    = wr(bodies[i].pos.y, ir, &vp) + 1;
        int c    = wc(bodies[i].pos.x, ic, &vp) + 1;
        int pair = i + 1;  /* CP_EARTH=1, CP_MOON=2, CP_SUN=3, CP_BODY4=4 */
        if (pair > 4) pair = CP_BODY4;
        if (WROW_OK(w, r)) {
            wattron(w, COLOR_PAIR(pair) | A_BOLD);
            mvwaddch(w, r, c, bodies[i].symbol);
            wattroff(w, COLOR_PAIR(pair) | A_BOLD);
        }
    }

    /* Spacecraft */
    {
        int r = wr(sc->pos.y, ir, &vp) + 1;
        int c = wc(sc->pos.x, ic, &vp) + 1;
        char ch = '*';
        for (int i = 0; i < n_bodies; i++) {
            if (r == wr(bodies[i].pos.y, ir, &vp)+1 &&
                c == wc(bodies[i].pos.x, ic, &vp)+1) { ch = '#'; break; }
        }
        if (WROW_OK(w, r)) {
            wattron(w, COLOR_PAIR(CP_CRAFT) | A_BOLD);
            mvwaddch(w, r, c, ch);
            wattroff(w, COLOR_PAIR(CP_CRAFT) | A_BOLD);
        }
    }

    wattron(w, COLOR_PAIR(CP_LABEL));
    mvwprintw(w, rows-1, 2, " x=L1  *=Craft  .=Trail ");
    wattroff(w, COLOR_PAIR(CP_LABEL));
    wrefresh(w);
}

/* ─── Telemetry window ───────────────────────────────────────── */

static void draw_data(WINDOW           *w,
                      const Body       *bodies, int n_bodies,
                      const Spacecraft *sc,
                      int               steps_frame, double dt,
                      int               paused, int recording,
                      long              total_steps)
{
    wclear(w);
    draw_box(w, "TELEMETRY");

    int row = 2, c1 = 2, c2 = 14;

#define LBL(s) do { if (WROW_OK(w,row)) { \
    wattron(w,COLOR_PAIR(CP_LABEL)); mvwprintw(w,row,c1,"%s",s); \
    wattroff(w,COLOR_PAIR(CP_LABEL)); } } while(0)
#define VAL(fmt,...) do { if (WROW_OK(w,row)) { \
    wattron(w,COLOR_PAIR(CP_VALUE)|A_BOLD); \
    mvwprintw(w,row,c2,fmt,__VA_ARGS__); \
    wattroff(w,COLOR_PAIR(CP_VALUE)|A_BOLD); } row++; } while(0)
#define SECT(s) do { row++; if (WROW_OK(w,row)) { \
    wattron(w,COLOR_PAIR(CP_TITLE)|A_UNDERLINE); \
    mvwprintw(w,row,c1,"%s",s); \
    wattroff(w,COLOR_PAIR(CP_TITLE)|A_UNDERLINE); } row++; } while(0)

    SECT("Time");
    LBL("Elapsed"); VAL("%.3f h",  sc->time);
    LBL("Dist");    VAL("%.0f km", sc->distance_traveled);
    LBL("Steps");   VAL("%ld",     total_steps);

    SECT("Position (km)");  /* relative to reference body */
    LBL("X"); VAL("%+.1f", sc->pos.x - bodies[0].pos.x);
    LBL("Y"); VAL("%+.1f", sc->pos.y - bodies[0].pos.y);

    SECT("Velocity");  /* relative to reference body */
    LBL("Vx");     VAL("%+.1f km/h",  sc->vel.x - bodies[0].vel.x);
    LBL("Vy");     VAL("%+.1f km/h",  sc->vel.y - bodies[0].vel.y);
    LBL("|V|");    VAL("%.1f km/h",   v2_len(v2_sub(sc->vel, bodies[0].vel)));
    LBL("km/s");   VAL("%.4f",        v2_len(v2_sub(sc->vel, bodies[0].vel))/3600.0);

    SECT("Accel (km/h\xc2\xb2)");
    LBL("Ax"); VAL("%+.5f", sc->acc.x);
    LBL("Ay"); VAL("%+.5f", sc->acc.y);

    SECT("Distances (km)");
    for (int i = 0; i < n_bodies; i++) {
        double d   = v2_dist(sc->pos, bodies[i].pos);
        int  alert = d < bodies[i].radius * 5.0;
        char lbl[20];
        snprintf(lbl, sizeof lbl, "%-10s", bodies[i].name);
        LBL(lbl);
        if (WROW_OK(w,row)) {
            wattron(w, COLOR_PAIR(alert ? CP_ALERT : CP_VALUE) | A_BOLD);
            mvwprintw(w, row, c2, "%.0f%s", d, alert ? " !" : "");
            wattroff(w, COLOR_PAIR(alert ? CP_ALERT : CP_VALUE) | A_BOLD);
        }
        row++;
    }

    SECT("Simulation");
    {
        double h_s = (double)steps_frame * dt * TARGET_FPS;
        LBL("h/s");      VAL("%.1f", h_s);
        LBL("Steps/fr"); VAL("%d",   steps_frame);
        LBL("dt (h)");   VAL("%.5f", dt);
    }

    row++;
    if (WROW_OK(w,row)) {
        wattron(w, COLOR_PAIR(paused ? CP_ALERT : CP_OK) | A_BOLD);
        mvwprintw(w, row++, c1, paused ? "[PAUSED ]" : "[RUNNING]");
        wattroff(w, COLOR_PAIR(paused ? CP_ALERT : CP_OK) | A_BOLD);
    }
    if (WROW_OK(w,row)) {
        wattron(w, COLOR_PAIR(recording ? CP_ALERT : CP_LABEL) | A_BOLD);
        mvwprintw(w, row++, c1, recording ? "[REC \xe2\x97\x8f]" : "[rec \xe2\x97\x8b]");
        wattroff(w, COLOR_PAIR(recording ? CP_ALERT : CP_LABEL) | A_BOLD);
    }

    int cr = getmaxy(w) - 9;
    if (cr < row) cr = row;
    if (WROW_OK(w, cr)) {
        wattron(w, COLOR_PAIR(CP_LABEL));
        mvwprintw(w, cr++, c1, "Controls:");
        mvwprintw(w, cr++, c1, " p  pause/resume");
        mvwprintw(w, cr++, c1, " +  faster  -  slower");
        mvwprintw(w, cr++, c1, " e  export CSV+plot");
        mvwprintw(w, cr++, c1, " t  toggle recording");
        mvwprintw(w, cr++, c1, " r  restart  q  quit");
        wattroff(w, COLOR_PAIR(CP_LABEL));
    }

#undef LBL
#undef VAL
#undef SECT

    wrefresh(w);
}

/* ─── Status bar ─────────────────────────────────────────────── */

static void draw_status(WINDOW *w, const char *msg, int alert)
{
    wattron(w, COLOR_PAIR(alert ? CP_ALERT : CP_LABEL) | A_REVERSE);
    mvwhline(w, 0, 0, ' ', getmaxx(w));
    mvwprintw(w, 0, 1, "%s", msg);
    wattroff(w, COLOR_PAIR(alert ? CP_ALERT : CP_LABEL) | A_REVERSE);
    wrefresh(w);
}

/* ═══════════════════════════════════════════════════════════════
 * Selection menus (in normal terminal, outside ncurses)
 * ═══════════════════════════════════════════════════════════════ */

static void print_header(void)
{
    printf("\n"
        "  \033[1;33m╔══════════════════════════════════════════════╗\033[0m\n"
        "  \033[1;33m║   SPACECRAFT TRAJECTORY SIMULATOR  v2.0     ║\033[0m\n"
        "  \033[1;33m╚══════════════════════════════════════════════╝\033[0m\n\n");
}

static int choose_scenario(void)
{
    const Scenario *sc = get_scenarios();
    printf("  \033[1;36mScenario:\033[0m\n\n");
    for (int i = 0; i < N_SCENARIOS; i++)
        printf("    \033[1m[%d]\033[0m  %-20s — %s\n",
               i+1, sc[i].name, sc[i].description);
    printf("\n  > ");
    fflush(stdout);
    int c = 1; if (scanf("%d", &c) != 1) c = 1;
    int ch; while ((ch = getchar()) != '\n' && ch != EOF) {}
    c--;
    return (c >= 0 && c < N_SCENARIOS) ? c : 0;
}

static int choose_mission(int scenario_idx)
{
    const Mission   *ms = get_missions();
    const Scenario  *sc = get_scenarios();
    printf("\n  \033[1;36mMission  [%s]:\033[0m\n\n", sc[scenario_idx].name);
    int shown = 0;
    for (int i = 0; i < N_MISSIONS; i++) {
        if (ms[i].scenario_idx != scenario_idx) continue;
        printf("    \033[1m[%d]\033[0m  %-18s %s\n"
               "         %s\n\n",
               i+1, ms[i].name, ms[i].year, ms[i].description);
        shown++;
    }
    if (!shown) printf("    (no presets for this scenario)\n");
    printf("    \033[1m[0]\033[0m  Custom\n\n  > ");
    fflush(stdout);
    int c = 0; if (scanf("%d", &c) != 1) c = 0;
    int ch; while ((ch = getchar()) != '\n' && ch != EOF) {}
    if (c >= 1 && c <= N_MISSIONS && ms[c-1].scenario_idx == scenario_idx)
        return c - 1;
    return N_MISSIONS;
}

/* ── SimConfig ───────────────────────────────────────────────── */

typedef struct {
    int         scenario_idx;
    double      x0, y0, vx0, vy0;
    double      dt, max_time;
    int         steps_frame;
    const char *mission_name;
    char        export_stem[64];
    int         export_decimation;
    int         export_flags;   /* EXPORT_IMAGES | EXPORT_VIDEO */
} SimConfig;

static void ask_double(const char *prompt, double *val) {
    printf("    %-42s [%.5g]: ", prompt, *val);
    fflush(stdout);
    char buf[64];
    if (fgets(buf, sizeof buf, stdin) && buf[0] != '\n') {
        double tmp; if (sscanf(buf, "%lf", &tmp) == 1) *val = tmp;
    }
}
static void ask_int(const char *prompt, int *val) {
    printf("    %-42s [%d]: ", prompt, *val);
    fflush(stdout);
    char buf[32];
    if (fgets(buf, sizeof buf, stdin) && buf[0] != '\n') {
        int tmp; if (sscanf(buf, "%d", &tmp) == 1) *val = tmp;
    }
}

static SimConfig build_config(int scenario_idx, int mission_idx)
{
    SimConfig cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.scenario_idx      = scenario_idx;
    cfg.export_decimation = 10;
    cfg.export_flags      = EXPORT_IMAGES | EXPORT_VIDEO;  /* both images and video by default */
    strncpy(cfg.export_stem, "trajectory", 63);

    if (mission_idx < N_MISSIONS) {
        const Mission *m = &get_missions()[mission_idx];
        cfg.x0           = m->x0;   cfg.y0  = m->y0;
        cfg.vx0          = m->vx0;  cfg.vy0 = m->vy0;
        cfg.dt           = m->dt;
        cfg.max_time     = m->max_time;
        cfg.steps_frame  = m->steps_frame;
        cfg.mission_name = m->name;
        snprintf(cfg.export_stem, 63, "%s_%s", m->name, m->year);
        for (char *p = cfg.export_stem; *p; p++) if (*p == ' ') *p = '_';
    } else {
        cfg.x0 = 7000; cfg.vy0 = 38500;
        cfg.dt = 0.02; cfg.max_time = 200; cfg.steps_frame = 100;
        cfg.mission_name = "Custom";
    }
    return cfg;
}

static void configure_interactive(SimConfig *cfg)
{
    def_prog_mode();
    endwin();
    print_header();

    int si = choose_scenario();
    int mi = choose_mission(si);
    *cfg   = build_config(si, mi);

    printf("\n  \033[1;36mParameters\033[0m  (Enter = keep default)\n\n");
    ask_double("Initial X (km)",          &cfg->x0);
    ask_double("Initial Y (km)",          &cfg->y0);
    ask_double("Initial Vx (km/h)",       &cfg->vx0);
    ask_double("Initial Vy (km/h)",       &cfg->vy0);
    printf("\n");
    ask_double("Max time (h)",            &cfg->max_time);
    ask_double("Time step dt (h)",        &cfg->dt);
    ask_int   ("Steps per frame",         &cfg->steps_frame);
    printf("\n");
    ask_int   ("Export: record every Nth step", &cfg->export_decimation);
    printf("    Export file stem                         [%s]: ",
           cfg->export_stem);
    fflush(stdout);
    char buf[64];
    if (fgets(buf, sizeof buf, stdin) && buf[0] != '\n') {
        buf[strcspn(buf, "\n")] = '\0';
        memcpy(cfg->export_stem, buf, 63); cfg->export_stem[63] = '\0';
    }
    printf("\n  Export media  (0 = data only | 1 = images | 2 = video | 3 = both)\n");
    ask_int ("  Generate on export", &cfg->export_flags);
    if (cfg->export_flags < 0 || cfg->export_flags > 3) cfg->export_flags = 0;

    /* Clamp */
    if (cfg->dt           <= 0) cfg->dt           = 0.001;
    if (cfg->max_time     <= 0) cfg->max_time     = 100;
    if (cfg->steps_frame   < 1) cfg->steps_frame  = 1;
    if (cfg->steps_frame > MAX_STEPS) cfg->steps_frame = MAX_STEPS;
    if (cfg->export_decimation < 1) cfg->export_decimation = 1;

    printf("\n  Launching \033[1;33m%s\033[0m in \033[1;36m%s\033[0m...\n",
           cfg->mission_name,
           get_scenarios()[cfg->scenario_idx].name);
    sleep_us(500000);
    reset_prog_mode();
    refresh();
}

/* ═══════════════════════════════════════════════════════════════
 * Live simulation loop
 * ═══════════════════════════════════════════════════════════════ */

typedef enum {
    SIM_RUNNING, SIM_PAUSED,
    SIM_DONE_TIMEOUT, SIM_DONE_COLLISION,
    SIM_QUIT, SIM_RESTART,
} SimState;

static int run_live(SimConfig *cfg)
{
    const Scenario *scenario = &get_scenarios()[cfg->scenario_idx];
    Body bodies[MAX_BODIES];
    int  n_bodies = scenario_load_bodies(scenario, bodies);

    Spacecraft sc;
    sc.pos               = v2_add(bodies[0].pos, v2(cfg->x0, cfg->y0));
    sc.vel               = v2_add(bodies[0].vel, v2(cfg->vx0, cfg->vy0));
    sc.acc               = resultant_acc(bodies, n_bodies, sc.pos);
    sc.distance_traveled = 0.0;
    sc.time              = 0.0;

    Trail trail = {0};
    trail_push(&trail, v2(0, 0));  /* initial pos relative to bodies[0] = (0,0) */

    Exporter ex;
    exporter_init(&ex, cfg->export_stem, cfg->export_decimation,
                  cfg->dt, cfg->mission_name, scenario->name);

    int map_cols  = (int)(COLS * 0.66);
    int data_cols = COLS - map_cols;
    int main_rows = LINES - 1;

    WINDOW *map_w    = newwin(main_rows, map_cols,  0, 0);
    WINDOW *data_w   = newwin(main_rows, data_cols, 0, map_cols);
    WINDOW *status_w = newwin(1, COLS, main_rows, 0);

    if (!map_w || !data_w || !status_w) {
        endwin(); fprintf(stderr, "Terminal too small.\n"); return 0;
    }

    nodelay(stdscr, TRUE);

    SimState state          = SIM_RUNNING;
    int      collision_body = -1;
    int      steps_frame    = cfg->steps_frame;
    long     total_steps    = 0;
    int      recording      = 1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (state != SIM_QUIT         &&
           state != SIM_DONE_TIMEOUT &&
           state != SIM_DONE_COLLISION &&
           state != SIM_RESTART)
    {
        /* Input */
        int ch;
        while ((ch = getch()) != ERR) {
            switch (ch) {
            case 'q': case 'Q': state = SIM_QUIT;    break;
            case 'r': case 'R': state = SIM_RESTART; break;
            case 'p': case 'P': case ' ':
                state = (state == SIM_RUNNING) ? SIM_PAUSED : SIM_RUNNING;
                break;
            case '+': case '=':
                steps_frame = steps_frame < 10 ? steps_frame*2
                                               : steps_frame + steps_frame/2;
                if (steps_frame > MAX_STEPS) steps_frame = MAX_STEPS;
                break;
            case '-': case '_':
                steps_frame = steps_frame > 2 ? steps_frame/2 : 1;
                break;
            case 'e': case 'E':
                if (exporter_write(&ex) == 0) {
                    char msg[256];
                    snprintf(msg, sizeof msg,
                        " Exported: %.60s.csv + %.60s.plt",
                        cfg->export_stem, cfg->export_stem);
                    draw_status(status_w, msg, 0);
                    wrefresh(status_w);
                    if (cfg->export_flags) {
                        /* Suspend ncurses while gnuplot / cc / ffmpeg run */
                        def_prog_mode();
                        endwin();
                        exporter_render(&ex, cfg->export_flags);
                        reset_prog_mode();
                        refresh();
                    }
                    sleep_us(2500000);
                }
                break;
            case 't': case 'T':
                recording = !recording;
                break;
            case KEY_RESIZE:
                wresize(map_w,    LINES-1, (int)(COLS*0.66));
                wresize(data_w,   LINES-1, COLS-(int)(COLS*0.66));
                wresize(status_w, 1,       COLS);
                mvwin(data_w,   0, (int)(COLS*0.66));
                mvwin(status_w, LINES-1, 0);
                break;
            }
        }

        /* Advance */
        if (state == SIM_RUNNING) {
            for (int i = 0; i < steps_frame && state == SIM_RUNNING; i++) {
                step_verlet(&sc, bodies, n_bodies, cfg->dt);
                step_bodies(bodies, n_bodies, cfg->dt);
                total_steps++;
                trail_push(&trail, v2_sub(sc.pos, bodies[0].pos));
                if (recording)
                    exporter_record(&ex, &sc, bodies, n_bodies);
                int hit = check_collision(&sc, bodies, n_bodies);
                if (hit >= 0) {
                    state = SIM_DONE_COLLISION;
                    collision_body = hit;
                } else if (sc.time >= cfg->max_time) {
                    state = SIM_DONE_TIMEOUT;
                }
            }
        }

        /* Render */
        draw_map(map_w, scenario, bodies, n_bodies, &sc, &trail);
        draw_data(data_w, bodies, n_bodies, &sc,
                  steps_frame, cfg->dt,
                  state == SIM_PAUSED, recording, total_steps);

        {
            char status[256];
            double h_s = (double)steps_frame * cfg->dt * TARGET_FPS;
            if (state == SIM_PAUSED)
                snprintf(status, sizeof status,
                    " PAUSED | %s | %.3f h | p:resume  e:export  r:restart  q:quit",
                    cfg->mission_name, sc.time);
            else
                snprintf(status, sizeof status,
                    " %s | %.3f h | %.1f h/s | %ld steps | "
                    "p:pause  +/-:speed  e:export  t:rec  r:restart  q:quit",
                    cfg->mission_name, sc.time, h_s, total_steps);
            draw_status(status_w, status, 0);
        }

        /* Frame timing */
        clock_gettime(CLOCK_MONOTONIC, &t1);
        long elapsed_us = (t1.tv_sec  - t0.tv_sec)  * 1000000L
                        + (t1.tv_nsec - t0.tv_nsec) / 1000L;
        if (elapsed_us < FRAME_US) sleep_us(FRAME_US - elapsed_us);
        clock_gettime(CLOCK_MONOTONIC, &t0);
    }

    /* End-of-run overlay */
    if (state == SIM_DONE_COLLISION || state == SIM_DONE_TIMEOUT) {
        nodelay(stdscr, FALSE);
        char msg[256];
        if (state == SIM_DONE_COLLISION)
            snprintf(msg, sizeof msg,
                " COLLISION: %s after %.3f h | %.0f km traveled"
                " | Auto-exporting… press any key",
                bodies[collision_body].name, sc.time, sc.distance_traveled);
        else
            snprintf(msg, sizeof msg,
                " Time limit reached (%.3f h) | Craft still in flight"
                " | Auto-exporting… press any key", sc.time);
        draw_status(status_w, msg, 1);
        exporter_write(&ex);
        if (cfg->export_flags) {
            def_prog_mode();
            endwin();
            exporter_render(&ex, cfg->export_flags);
            reset_prog_mode();
            refresh();
        }
        getch();
    }

    exporter_free(&ex);
    delwin(map_w);
    delwin(data_w);
    delwin(status_w);

    return (state == SIM_RESTART) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Batch mode
 * ═══════════════════════════════════════════════════════════════ */

static void batch_process(FILE *fp)
{
    const Scenario *scenarios = get_scenarios();
    const Scenario *sc = &scenarios[0];  /* Earth-Moon default */
    Body bodies[MAX_BODIES];
    int  n_bodies = scenario_load_bodies(sc, bodies);

    char   code;
    double xA, yA, mA, xB, yB, mB, x, y, vx, vy, dt, max_time;

    while (fscanf(fp, " %c", &code) == 1) {
        printf("─── [%c] ──────────────────────────────────────\n", code);
        switch (code) {

        case 'a': {
            if (fscanf(fp, "%lf%lf%lf%lf%lf", &xA,&yA,&mA,&x,&y) != 5) break;
            Body b = { .pos={xA,yA}, .mass=mA };
            Vec2 a = grav_acc(&b, v2(x, y));
            printf("  grav. accel. : (%.6f, %.6f) km/h²\n\n", a.x, a.y);
            break;
        }
        case 'A': {
            if (fscanf(fp, "%lf%lf%lf%lf%lf%lf%lf%lf",
                       &xA,&yA,&mA,&xB,&yB,&mB,&x,&y) != 8) break;
            Body b1={.pos={xA,yA},.mass=mA}, b2={.pos={xB,yB},.mass=mB};
            Body two[2]={b1,b2};
            Vec2 a1=grav_acc(&b1,v2(x,y)), a2=grav_acc(&b2,v2(x,y));
            Vec2 at=resultant_acc(two,2,v2(x,y));
            printf("  body1: (%.6f, %.6f)\n  body2: (%.6f, %.6f)\n"
                   "  total: (%.6f, %.6f) km/h²\n\n",
                   a1.x,a1.y, a2.x,a2.y, at.x,at.y);
            break;
        }
        case 'e': {
            if (fscanf(fp, "%lf%lf%lf%lf%lf", &xA,&yA,&mA,&x,&y) != 5) break;
            Body b={.pos={xA,yA},.mass=mA};
            printf("  dist=%.2f km  v_esc=%.2f km/h\n\n",
                   v2_dist(v2(xA,yA),v2(x,y)), escape_velocity(&b,v2(x,y)));
            break;
        }
        case 'L': {
            if (fscanf(fp,"%lf%lf%lf%lf%lf%lf",&xA,&yA,&mA,&xB,&yB,&mB)!=6) break;
            Body b1={.pos={xA,yA},.mass=mA}, b2={.pos={xB,yB},.mass=mB};
            Vec2 l1=lagrange_l1(&b1,&b2);
            printf("  L1 = (%.2f, %.2f) km\n\n", l1.x, l1.y);
            break;
        }
        case 's': {
            int scenario_idx = 0;  /* default: Earth-Moon */
            char mission_name[128] = "";  /* mission identifier */
            /* Parse: s scenario_idx mission_name x y vx vy max_time dt [flags] */
            if (fscanf(fp,"%d%127s%lf%lf%lf%lf%lf%lf",
                       &scenario_idx,mission_name,&x,&y,&vx,&vy,&max_time,&dt)!=8) break;
            
            /* Validate scenario index */
            if (scenario_idx < 0 || scenario_idx >= N_SCENARIOS) scenario_idx = 0;
            sc = &scenarios[scenario_idx];
            n_bodies = scenario_load_bodies(sc, bodies);
            
            int flags = EXPORT_IMAGES | EXPORT_VIDEO;   /* default: both */
            { int r; if (fscanf(fp, " %d", &r) == 1) flags = r; }   /* optional override */
            if (flags < 0 || flags > 3) flags = EXPORT_IMAGES | EXPORT_VIDEO;
            Spacecraft craft;
            craft.pos = v2_add(bodies[0].pos, v2(x, y));
            craft.vel = v2_add(bodies[0].vel, v2(vx, vy));
            craft.acc=resultant_acc(bodies,n_bodies,craft.pos);
            craft.distance_traveled=0; craft.time=0;

            Exporter ex;
            exporter_init(&ex,"batch_trajectory",20,dt,mission_name,sc->name);

            int coll=-1;
            while (craft.time < max_time) {
                step_verlet(&craft,bodies,n_bodies,dt);
                step_bodies(bodies,n_bodies,dt);
                exporter_record(&ex,&craft,bodies,n_bodies);
                coll=check_collision(&craft,bodies,n_bodies);
                if (coll>=0) break;
            }
            exporter_write(&ex);
            if (flags) exporter_render(&ex, flags);
            exporter_free(&ex);

            printf("  time=%.4f h  dist=%.0f km  pos=(%.2f,%.2f)\n",
                   craft.time, craft.distance_traveled,
                   craft.pos.x - bodies[0].pos.x,
                   craft.pos.y - bodies[0].pos.y);
            printf("  result: %s\n",
                   coll>=0 ? bodies[coll].name : "still in flight");
            printf("  files: batch_trajectory.csv + batch_trajectory.plt"
                   "%s%s\n\n",
                   flags & EXPORT_IMAGES ? " + images" : "",
                   flags & EXPORT_VIDEO  ? " + video"  : "");
            break;
        }
        default:
            printf("  (unknown '%c')\n\n", code);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * Entry point
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc > 1) {
        FILE *fp = fopen(argv[1], "r");
        if (!fp) { perror(argv[1]); return 1; }
        printf("Spacecraft Trajectory Simulator — Batch Mode\n\n");
        batch_process(fp);
        fclose(fp);
        return 0;
    }

    initscr();
    if (!has_colors()) { endwin(); fputs("No colour support.\n",stderr); return 1; }
    if (COLS < 60 || LINES < 20) {
        endwin();
        fprintf(stderr,"Terminal too small (%dx%d). Need 60x20+.\n",COLS,LINES);
        return 1;
    }

    init_colors();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(0);

    SimConfig cfg;
    memset(&cfg, 0, sizeof cfg);

    int restart;
    do {
        configure_interactive(&cfg);
        clear(); refresh();
        restart = run_live(&cfg);
    } while (restart);

    endwin();
    return 0;
}
