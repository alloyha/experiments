# Spacecraft Trajectory Simulator
#
# Directory structure:
#   src/       — source files (.c, .h)
#   build/     — compiled object files (.o)
#   bin/       — executables (simulator, test_physics)
#   missions/output/  — simulation outputs (CSV, PLT, PNG, MP4, etc.)
#   tests/     — test support files
#
# Usage:
#   make              → build bin/simulator
#   make test         → build & run bin/test_physics
#   make all          → both
#   make clean        → remove build/ and bin/ (keep missions/output)
#   make clean-all    → also remove missions/output and generated sim files

CC       := gcc
CFLAGS   := -Wall -Wextra -O2 -std=c11 -D_POSIX_C_SOURCE=200809L
LDFLAGS  := -lncurses -lm

# Directories
SRC_DIR  := src
BUILD_DIR:= build
BIN_DIR  := bin
OUT_DIR  := missions/output

# Source and object files
SIM_SRCS  := $(addprefix $(SRC_DIR)/,main.c physics.c bodies.c missions.c export.c)
TEST_SRCS := $(addprefix $(SRC_DIR)/,test.c physics.c bodies.c)
HEADERS   := $(addprefix $(SRC_DIR)/,physics.h bodies.h missions.h export.h)

SIM_OBJS  := $(addprefix $(BUILD_DIR)/,$(notdir $(SIM_SRCS:.c=.o)))
TEST_OBJS := $(addprefix $(BUILD_DIR)/,$(notdir $(TEST_SRCS:.c=.o)))

# Executables
SIMULATOR := $(BIN_DIR)/simulator
TEST_BIN  := $(BIN_DIR)/test_physics

.PHONY: all clean clean-all test simulator help vars
.PHONY: run run-all list

# Help target
help:
	@echo "Spacecraft Trajectory Simulator — Build targets:"
	@echo ""
	@echo "  make              Build simulator to bin/simulator"
	@echo "  make test         Build & run tests in bin/test_physics"
	@echo "  make all          Build simulator and test_physics"
	@echo "  make clean        Remove build/ and bin/"
	@echo "  make clean-all    clean + remove missions/output/ and simulation files"
	@echo "  make help         Show this message"
	@echo ""
	@echo "Mission simulation:"
	@echo "  make list              Show available mission presets"
	@echo "  make run [PARAMS]      Run simulation with parameters"
	@echo "  make run-all           Run all 7 preset missions"
	@echo ""
	@echo "Directory layout:"
	@echo "  src/              Source files (.c, .h)"
	@echo "  build/            Compiled objects (.o)"
	@echo "  bin/              Executables"
	@echo "  missions/output/  Simulation outputs (CSV, PNG, MP4)"
	@echo "  tests/            Test-related files"

# Default target: build simulator only
simulator: $(SIMULATOR)

# Build both simulator and test suite
all: $(SIMULATOR) $(TEST_BIN)

# Simulator executable
$(SIMULATOR): $(SIM_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Test executable
$(TEST_BIN): $(TEST_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Object file compilation rules
# Explicit rule for test.o to avoid redundant header dependencies
$(BUILD_DIR)/test.o: $(SRC_DIR)/test.c $(SRC_DIR)/physics.h $(SRC_DIR)/bodies.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Generic rule for other .o files from .c
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Create directories if they don't exist
$(BUILD_DIR) $(BIN_DIR) $(OUT_DIR):
	@mkdir -p $@

# Test target — build and run
test: $(TEST_BIN)
	./$<

# ═══════════════════════════════════════════════════════════════
# SIMULATION TARGETS
# ═══════════════════════════════════════════════════════════════

# Default parameters for generic 'run' target (Apollo 11)
# Moon starts at (+384400, 0); craft starts at (−LEO_R, 0) so Hohmann apogee
# falls at the Moon.  Velocity is prograde (+y = counterclockwise).
mission := apollo11
scenario := 0
x := -6556
y := 0
vx := 0
vy := 39038
max_time := 120
dt := 0.02

# Generic run target with simulation parameters
# Usage: make run [mission=name] [scenario=0] [x=6556] [y=0] [vx=0] [vy=39038] [max_time=120] [dt=0.02]
# Generates: CSV + PNG + MP4, then cleans up temporary files
.PHONY: run
run: $(SIMULATOR)
	@echo "════════════════════════════════════════════"
	@echo "Running simulation with parameters:"
	@echo "  scenario:  $(scenario)"
	@echo "  position:  ($(x), $(y)) km"
	@echo "  velocity:  ($(vx), $(vy)) km/h"
	@echo "  max_time:  $(max_time) h"
	@echo "  dt:        $(dt) h"
	@echo "════════════════════════════════════════════"
	@printf 's $(scenario) $(mission) $(x) $(y) $(vx) $(vy) $(max_time) $(dt)\n' | $(SIMULATOR) /dev/stdin
	@echo ""
	@echo "Generating plots..."
	@MISSION_DIR="$$(find missions/output -mindepth 1 -maxdepth 1 -type d -printf '%T@\t%p\n' | sort -rn | head -1 | cut -f2)"; \
	if [ -z "$$MISSION_DIR" ] || [ ! -d "$$MISSION_DIR" ]; then \
		echo "Error: No mission output directory found"; \
		exit 1; \
	fi; \
	MISSION_NAME="$$(basename $$MISSION_DIR)"; \
	PLT="$$(find $$MISSION_DIR -maxdepth 1 -type f -name "*.plt" | head -1)"; \
	if [ -n "$$PLT" ] && command -v gnuplot >/dev/null 2>&1; then \
		echo "✓ Generating PNG images with gnuplot"; \
		PLT_BASE="$$(basename $$PLT)"; \
		sed -e 's/set terminal pngcairo.*/set terminal png size 1200,900/' \
		    -e "s|'missions/output/[^/]*/|'|g" "$$PLT" > "$$MISSION_DIR/$${PLT_BASE}.tmp" && \
		(cd "$$MISSION_DIR" && gnuplot "$${PLT_BASE}.tmp" 2>&1 | grep -v "^$$" | head -5 || true) && \
		rm -f "$$MISSION_DIR/$${PLT_BASE}.tmp"; \
	fi; \
	echo "Generating animation..."; \
	ANIMATOR="$$(find $$MISSION_DIR -maxdepth 1 -type f -name "*_animate.c" | head -1)"; \
	if [ -n "$$ANIMATOR" ]; then \
		echo "✓ Compiling animator..."; \
		cc -O2 -o $$MISSION_DIR/animate $$ANIMATOR -lm 2>&1 | head -3 || true; \
		echo "✓ Generating video..."; \
		cd $$MISSION_DIR && ./animate 2>&1 | grep -E "Rendering|Done" | head -2; \
		cd - > /dev/null; \
		if [ -f $$MISSION_DIR/batch_trajectory.mp4 ]; then \
			find $$MISSION_DIR -maxdepth 1 -type f \( -name "*_animate.c" -o -name "animate" \) -delete 2>/dev/null || true; \
			find $$MISSION_DIR -maxdepth 1 -type f -name "*.sh" -delete 2>/dev/null || true; \
		fi; \
	fi; \
	find $$MISSION_DIR -maxdepth 1 -type f -name "*.plt" -delete 2>/dev/null || true; \
	echo ""; \
	echo "✓ Mission complete: $$MISSION_NAME"; \
	echo "  Generated outputs:"; \
	ls -lh $$MISSION_DIR 2>/dev/null | grep -E "csv|mp4|png" | awk '{print "    " $$9 " (" $$5 ")"}' || echo "    (no outputs found)"

# Run all 7 preset missions
.PHONY: run-all
run-all:
	@echo "Running all 7 preset missions..."
	@$(MAKE) run mission=apollo11 scenario=0 x=-6556 y=0 vx=0 vy=39038 max_time=120 dt=0.02
	@echo ""
	@$(MAKE) run mission=apollo13 scenario=0 x=-6556 y=0 vx=0 vy=38650 max_time=200 dt=0.02
	@echo ""
	@$(MAKE) run mission=artemis1 scenario=0 x=-6556 y=0 vx=0 vy=36756 max_time=300 dt=0.02
	@echo ""
	@$(MAKE) run mission=earth-escape scenario=1 x=-6556 y=0 vx=0 vy=42000 max_time=600 dt=0.05
	@echo ""
	@$(MAKE) run mission=moon-impact scenario=0 x=6556 y=0 vx=49800 vy=4500 max_time=12 dt=0.005
	@echo ""
	@$(MAKE) run mission=mars-approach scenario=2 x=3889.5 y=0 vx=0 vy=12600 max_time=30 dt=0.002
	@echo ""
	@$(MAKE) run mission=jupiter-io scenario=3 x=80000 y=200000 vx=80000 vy=-50000 max_time=5 dt=0.001
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "✓ All 7 missions completed successfully"
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Mission outputs:"
	@ls -d missions/output/*/ 2>/dev/null | while read dir; do echo "  - $$(basename $$dir)"; done || echo "  (no missions completed)"

# List available presets and usage examples
.PHONY: list
list:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Available Mission Presets & Usage"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "QUICK START:"
	@echo "  make run                # Apollo 11 (default)"
	@echo "  make run-all            # All 7 missions"
	@echo ""
	@echo "PRESET MISSIONS:"
	@echo ""
	@echo "  Apollo 11           : make run mission=apollo11 scenario=0 x=-6556 y=0 vx=0 vy=39038 max_time=120 dt=0.02"
	@echo "  Apollo 13           : make run mission=apollo13 scenario=0 x=-6556 y=0 vx=0 vy=38650 max_time=200 dt=0.02"
	@echo "  Artemis I           : make run mission=artemis1 scenario=0 x=-6556 y=0 vx=0 vy=36756 max_time=300 dt=0.02"
	@echo "  Earth Escape        : make run mission=earth-escape scenario=1 x=-6556 y=0 vx=0 vy=42000 max_time=600 dt=0.05"
  @echo "  Moon Impact         : make run mission=moon-impact scenario=0 x=6556 y=0 vx=49800 vy=4500 max_time=12 dt=0.005"
	@echo "  Mars Approach       : make run mission=mars-approach scenario=2 x=3889.5 y=0 vx=0 vy=12600 max_time=30 dt=0.002"
	@echo "  Jupiter-Io Flyby    : make run mission=jupiter-io scenario=3 x=80000 y=200000 vx=80000 vy=-50000 max_time=5 dt=0.001"
	@echo ""
	@echo "SCENARIO CODES:"
	@echo "  0 = Earth-Moon"
	@echo "  1 = Earth-Moon-Sun"
	@echo "  2 = Mars-Phobos"
	@echo "  3 = Jupiter-Io"
	@echo ""
	@echo "EACH MISSION GENERATES:"
	@echo "  ✓ CSV trajectory data"
	@echo "  ✓ PNG telemetry plots (3 plots per mission)"
	@echo "  ✓ MP4 video animation"
	@echo ""
	@echo "CUSTOM SIMULATION:"
	@echo "  make run scenario=<0-3> x=<km> y=<km> vx=<km/h> vy=<km/h> max_time=<h> dt=<h>"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make run                              # Use Apollo 11 defaults"
	@echo "  make run scenario=2 max_time=50      # Mars scenario, 50 hour sim"
	@echo "  make run vy=30000 max_time=200       # Custom velocity, 200 hours"

# Clean built objects and executables (keeps mission outputs)
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Clean everything including mission outputs
clean-all: clean
	rm -rf $(OUT_DIR)/*
	rm -f *.csv *.plt *.png *.mp4 *_animate.c *_animate 2>/dev/null || true

# Print Makefile variables (for debugging)
vars:
	@echo "SIM_SRCS: $(SIM_SRCS)"
	@echo "SIM_OBJS: $(SIM_OBJS)"
	@echo "TEST_OBJS: $(TEST_OBJS)"
	@echo "SIMULATOR: $(SIMULATOR)"
	@echo "TEST_BIN: $(TEST_BIN)"
