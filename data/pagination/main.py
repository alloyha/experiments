from strategies import benchmarks
from database import populate_database
from utils import display_results, run_all_benchmarks

if __name__ == "__main__":
    IS_POPULATE = False

    if IS_POPULATE:
        populate_database()

    run_all_benchmarks(benchmarks)
    display_results(benchmarks)


    