import os
from time import time
from typing import Callable, List, Tuple

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import logger

from tabulate import tabulate
import df2img
import logging

ResultRow = Tuple[int | str, float]
Benchmark = List[Tuple[str, Callable]]

# Create a directory to store results if it doesn't already exist
def save_results(name, results):
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results, columns=["cursor", "elapsed_seconds"])
    df.to_csv(f"results/{name}_benchmark.csv", index=False)

# Display the descriptive statistics and sort by the median ('50%' column)
def display_results(benchmarks):
    strategies_results = []

    # Read each strategy result and append it
    for name, method in benchmarks:
        file_path = f"results/{name}_benchmark.csv"
        if os.path.exists(file_path):
            strategy_result = pd.read_csv(file_path)
            strategy_result['strategy'] = name
            strategies_results.append(strategy_result)
        else:
            logger.warning(f"File {file_path} does not exist. Skipping.")

    if not strategies_results:
        logger.error("No benchmark results found. Exiting function.")
        return

    # Combine all results into a single DataFrame
    df = pd.concat(strategies_results, ignore_index=True)

    # Calculate descriptive statistics, including 95th and 99th percentiles
    percentiles = [0.25, 0.5, 0.75, 0.95, 0.99]
    summary = df.groupby('strategy')['elapsed_seconds'].describe(percentiles=percentiles).round(4)

    # Sort the summary by the '99%' column
    summary_sorted = summary.sort_values(by='99%', ascending=True)

    # Display the summary using tabulate
    logger.info("📊 Descriptive Statistics by Strategy:")
    print(tabulate(summary_sorted, headers='keys', tablefmt='grid'))

    # Save the summary as an image
    try:
        fig = df2img.plot_dataframe(
            summary_sorted.reset_index(),
            title=dict(
                text="Benchmark Summary Statistics",
                font=dict(size=20),
                x=0.5,
                xanchor="center"
            ),
            row_fill_color=("#f5f5f5", "white"),
            col_width=[2] + [1]*len(summary_sorted.columns),
            fig_size=(800, 400)
        )
        df2img.save_dataframe(fig=fig, filename="results/benchmark_summary.png")
        logger.info("✅ Summary image saved to results/benchmark_summary.png")
    except Exception as e:
        logger.error(f"Failed to save summary image: {e}")

# Run the individual benchmark strategy and save results
def run_strategy(name, method):
    try:
        logger.info(f"⏱️ {name}...")
        start_time = time()
        results = method()
        elapsed_time = time() - start_time
        save_results(name, results)
        logger.info(f"✅ {name} benchmark completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"❌ {name} benchmark failed: {e}")
    return name

def run_all_benchmarks(benchmarks: Benchmark) -> None:
    overall_start = time()

    with ThreadPoolExecutor(max_workers=len(benchmarks)) as executor:
        future_to_name = {
            executor.submit(run_strategy, name, fn): name
            for name, fn in benchmarks
        }

        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
            except Exception as e:
                print(f"🔥 Unexpected error in {name}: {e}")
    
    print(f"🏁 All benchmarks finished in {time() - overall_start:.2f} seconds.")