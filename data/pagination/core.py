import os
import sys
from time import time
from typing import Callable, List, Tuple
import contextlib
import logging

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from tabulate import tabulate
from df2img import plot_dataframe, save_dataframe



from config import logger, TOTAL_ROWS, LIMIT
from strategies import run_paginated

ResultRow = Tuple[int | str, float]
Benchmark = List[Tuple[str, Callable]]

# Create a directory to store results if it doesn't already exist
def save_results(name, results):
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results, columns=["cursor", "elapsed_seconds"])
    df.to_csv(f"results/{name}_benchmark.csv", index=False)

def load_results(benchmarks):
    strategies_results = []

    for name, _ in benchmarks:
        file_path = f"results/{name}_benchmark.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['strategy'] = name
            strategies_results.append(df)
        else:
            logger.warning(f"⚠️ File not found: {file_path}. Skipping.")

    if not strategies_results:
        logger.error("🚨 No benchmark results loaded.")
        return None

    return pd.concat(strategies_results, ignore_index=True)


def summarize_results(df):
    percentiles = [0.25, 0.5, 0.75, 0.95, 0.99]
    summary = (
        df.groupby('strategy')['elapsed_seconds']
        .describe(percentiles=percentiles)
        .round(4)
        .sort_values(by='99%', ascending=True)
    )
    return summary


def display_summary(summary):
    logger.info("📊 Descriptive Statistics by Strategy:")
    print(tabulate(summary, headers="keys", tablefmt="grid"))

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def save_summary_image(summary, filename="results/benchmark_summary.png"):
    try:
        logger.info("Generating summary image")
        with suppress_stdout():
            fig = plot_dataframe(
                summary.reset_index(),
                title=dict(
                    text="Benchmark Summary Statistics",
                    font=dict(size=20),
                    x=0.5,
                    xanchor="center"
                ),
                row_fill_color=("#f5f5f5", "white"),
                col_width=[2] + [1]*len(summary.columns),
                fig_size=(800, 400)
            )

            _ = save_dataframe(fig=fig, filename=filename)
            logger.info(f"✅ Summary image saved to {filename}")
    except Exception as e:
        logger.error(f"❌ Failed to save summary image: {e}")

def display_results(benchmarks):
    df = load_results(benchmarks)
    if df is None:
        return
    summary = summarize_results(df)
    display_summary(summary)
    save_summary_image(summary)

def run_strategy(name, strategy_fn):
    try:
        logger.info(f"⏱️ Running {name} benchmark...")
        start_time = time()
        results = run_paginated(limit=LIMIT, total=TOTAL_ROWS, fetch_fn=strategy_fn)
        elapsed_time = time() - start_time
        save_results(name, results)  # Adaptar se você quiser salvar de forma diferente
        logger.info(f"✅ {name} completed in {elapsed_time:.2f}s")
    except Exception as e:
        logger.error(f"❌ {name} failed: {e}")
    return name


def run_all_benchmarks(benchmarks):
    overall_start = time()

    with ThreadPoolExecutor(max_workers=len(benchmarks)) as executor:
        futures = {
            executor.submit(run_strategy, name, fn): name
            for name, fn in benchmarks
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"🔥 Unexpected error in {name}: {e}")

    logger.info(f"🏁 All benchmarks finished in {time() - overall_start:.2f}s")