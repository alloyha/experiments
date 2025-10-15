# Python 3.14 (π Release) Free-Threading Benchmark Environment

A containerized environment for testing Python 3.14's free-threading capabilities with comprehensive round-robin producer-consumer queue benchmarks.

## 🚀 Quick Start

### Prerequisites
- Docker (20.10+)
- Docker Compose (v2.0+)

### Setup

1. **Place your files in the project directory:**
   ```
   project/
   ├── docker-compose.yml
   ├── Dockerfile (optional, for custom build)
   ├── Makefile
   ├── gil_roundrobin_bench.py
   └── README.md
   ```

2. **Run the benchmark with GIL disabled:**
   ```bash
   make run-nogil
   ```

3. **Compare GIL vs No-GIL:**
   ```bash
   make run-both
   ```

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `make run-nogil` | Run benchmark with GIL disabled (free-threading) |
| `make run-gil` | Run benchmark with GIL enabled (traditional) |
| `make run-both` | Run both configurations for comparison |
| `make compare` | Run comprehensive comparison and save results |
| `make dev` | Start interactive development container |
| `make multicore` | Run multi-core scaling test |
| `make test` | Quick test to verify Python 3.14 installation |
| `make shell` | Open bash shell in container |
| `make build` | Build custom Python 3.14 image from source |
| `make clean` | Remove all containers and images |

## 🐳 Docker Compose Services

### `python314-nogil`
Runs benchmark with GIL disabled for true parallelism
- Environment: `PYTHON_GIL=0`
- CPU: 4 cores
- Memory: 2GB

### `python314-gil`
Runs benchmark with GIL enabled for comparison
- Environment: `PYTHON_GIL=1`
- CPU: 4 cores
- Memory: 2GB

### `python314-dev`
Interactive development environment
- Bash shell access
- Mount workspace directory
- Full Python 3.14 tooling

### `python314-multicore`
Scales test across multiple core configurations
- Tests: 2, 4, and 8 cores
- Memory: 4GB
- Comprehensive scaling analysis

## 🎯 Usage Examples

### Basic Benchmark Run
```bash
# Using Docker Compose directly
docker-compose run --rm python314-nogil

# Using Make (recommended)
make run-nogil
```

### Interactive Development
```bash
make dev
# Inside container:
python gil_roundrobin_bench.py
python --version
```

### Custom Parameters
```bash
docker-compose run --rm python314-nogil python -c "
from gil_roundrobin_bench import run_benchmark
run_benchmark(num_consumers=8, num_items=20000)
"
```

### Build Custom Image
```bash
# Build Python 3.14 from source with optimizations
make build

# Update docker-compose.yml to use custom image:
# image: python314-nogil:latest
```

## 📊 Expected Results

### With GIL Disabled (Free-Threading)
- **Work Stealing**: Best performance (~12,000-15,000 items/sec)
- **Sharded Queues**: Close second (~11,000-14,000 items/sec)
- **True parallelism**: All cores utilized simultaneously
- **Lower latency variance**: More predictable performance

### With GIL Enabled (Traditional)
- **Similar performance across strategies**: (~8,000-10,000 items/sec)
- **Single-threaded execution**: Despite multiple threads
- **Higher latency variance**: Thread switching overhead

## 🔧 Customization

### Adjust Resources
Edit `docker-compose.yml`:
```yaml
services:
  python314-nogil:
    cpu_count: 8  # Increase CPU cores
    mem_limit: 4g # Increase memory
```

### Modify Benchmark Parameters
Edit the benchmark script or pass parameters:
```bash
docker-compose run --rm python314-nogil python -c "
from gil_roundrobin_bench import run_benchmark
run_benchmark(num_consumers=16, num_items=100000)
"
```

### Enable Profiling
```bash
make profile
# View results
python -m pstats profile.stats
```

## 🐛 Troubleshooting

### Issue: Python 3.14 not available
**Solution**: Python 3.14 was released in October 2024. Ensure you're using the latest official image or build from source using the provided Dockerfile.

### Issue: GIL status shows "ENABLED" when it should be disabled
**Solution**: Verify the `PYTHON_GIL=0` environment variable is set in docker-compose.yml

### Issue: Performance not improving with GIL disabled
**Solution**: 
- Ensure CPU-bound work (the benchmark uses SHA-256 hashing)
- Check CPU allocation: `docker stats`
- Verify multiple cores are available to the container

### Issue: Container exits immediately
**Solution**: Check logs with `make logs` or `docker-compose logs`

## 📚 Additional Resources

- [Python 3.14 Release Notes](https://docs.python.org/3.14/whatsnew/3.14.html)
- [PEP 703 - Making the GIL Optional](https://peps.python.org/pep-0703/)
- [Free-Threading Design](https://docs.python.org/3.14/whatsnew/3.13.html#free-threaded-cpython)

## 🎉 Celebrating Python 3.14 (π Release)

This benchmark celebrates the revolutionary π (3.14) release of Python, which introduces:
- **Free-threading**: Optional GIL removal for true parallelism
- **Performance improvements**: Up to 40% faster for parallel workloads
- **Backward compatibility**: Existing code runs without modifications

Happy benchmarking! 🐍🚀