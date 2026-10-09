from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_DATA_DIR = Path("data/KuaiRec/data")
DEFAULT_CACHE_DIR = Path("data/cache")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def prepare_kuairec_parquet(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)

    source = data_dir / "big_matrix.csv"
    target = cache_dir / "big_matrix.parquet"

    if not source.exists():
        raise FileNotFoundError(f"KuaiRec source not found: {source.resolve()}")

    if target.exists():
        print(f"Using Parquet cache: {target}")
        return target

    source_sql = _sql_path(source)
    target_sql = _sql_path(target)

    print(f"Creating Parquet cache from {source}...")

    con = duckdb.connect()

    try:
        con.execute(
            f"""
            COPY (
                SELECT
                    CAST(user_id AS INTEGER) AS user_id,
                    CAST(video_id AS INTEGER) AS video_id,
                    CAST(play_duration AS DOUBLE) AS play_duration,
                    CAST(video_duration AS DOUBLE) AS video_duration,
                    CAST(timestamp AS BIGINT) AS timestamp,
                    CAST(watch_ratio AS DOUBLE) AS watch_ratio
                FROM read_csv_auto(
                    '{source_sql}',
                    header = true
                )
            )
            TO '{target_sql}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000
            )
            """
        )
    except Exception:
        if target.exists():
            target.unlink()
        raise
    finally:
        con.close()

    print(f"Parquet cache created: {target}")

    return target


def load_interactions(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_users: int | None = None,
    min_watch_ratio: float | None = None,
    sampling: str = "random",
) -> pd.DataFrame:
    path = prepare_kuairec_parquet(
        data_dir=data_dir,
        cache_dir=cache_dir,
    )

    parquet_sql = _sql_path(path)

    predicates: list[str] = []

    if min_watch_ratio is not None:
        predicates.append(f"watch_ratio >= {float(min_watch_ratio)}")

    where_sql = ""

    if predicates:
        where_sql = "WHERE " + " AND ".join(predicates)

    con = duckdb.connect()

    try:
        if max_users is None:
            query = f"""
                SELECT
                    user_id,
                    video_id,
                    play_duration,
                    video_duration,
                    timestamp,
                    watch_ratio
                FROM read_parquet('{parquet_sql}')
                {where_sql}
            """

        else:
            max_users = int(max_users)

            if sampling == "random":
                selected_users_sql = f"""
                    SELECT user_id
                    FROM (
                        SELECT DISTINCT user_id
                        FROM read_parquet('{parquet_sql}')
                        {where_sql}
                    )
                    ORDER BY hash(user_id)
                    LIMIT {max_users}
                """

            elif sampling == "active":
                selected_users_sql = f"""
                    SELECT user_id
                    FROM read_parquet('{parquet_sql}')
                    {where_sql}
                    GROUP BY user_id
                    ORDER BY COUNT(*) DESC, user_id
                    LIMIT {max_users}
                """

            else:
                raise ValueError("sampling must be 'random' or 'active'")

            outer_filter = ""

            if predicates:
                outer_filter = "WHERE " + " AND ".join(
                    f"i.{predicate}" for predicate in predicates
                )

            query = f"""
                WITH selected_users AS (
                    {selected_users_sql}
                )
                SELECT
                    i.user_id,
                    i.video_id,
                    i.play_duration,
                    i.video_duration,
                    i.timestamp,
                    i.watch_ratio
                FROM read_parquet('{parquet_sql}') AS i
                INNER JOIN selected_users AS u
                    ON i.user_id = u.user_id
                {outer_filter}
            """

        df = con.execute(query).df()

    finally:
        con.close()

    if df.empty:
        raise RuntimeError("No interactions returned.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    return df


def temporal_split(
    df: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    result = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    sizes = result.groupby("user_id")["user_id"].transform("size")

    result = result[sizes >= min_interactions].copy()

    position = result.groupby("user_id").cumcount()

    sizes = result.groupby("user_id")["user_id"].transform("size")

    cutoff = (sizes * (1 - test_fraction)).astype(int)

    cutoff = cutoff.clip(lower=1)

    cutoff = pd.concat(
        [
            cutoff,
            sizes - 1,
        ],
        axis=1,
    ).min(axis=1)

    train = result[position < cutoff].copy().reset_index(drop=True)

    test = result[position >= cutoff].copy().reset_index(drop=True)

    return train, test


def sessionize(
    df: pd.DataFrame,
    *,
    session_gap_minutes: int = 30,
    max_session_items: int = 15,
) -> pd.DataFrame:
    result = df.sort_values(["user_id", "timestamp"]).copy()

    previous = result.groupby("user_id")["timestamp"].shift()

    gap = result["timestamp"] - previous

    new_session = previous.isna() | (gap > pd.Timedelta(minutes=session_gap_minutes))

    result["_natural_session"] = (
        new_session.groupby(result["user_id"]).cumsum().astype("int64")
    )

    position = result.groupby(
        [
            "user_id",
            "_natural_session",
        ]
    ).cumcount()

    result["_chunk"] = position // max_session_items

    result["context_id"] = (
        result["user_id"].astype(str)
        + ":"
        + result["_natural_session"].astype(str)
        + ":"
        + result["_chunk"].astype(str)
    )

    return result.drop(
        columns=[
            "_natural_session",
            "_chunk",
        ]
    ).reset_index(drop=True)


def describe_context_complexity(
    df: pd.DataFrame,
) -> None:
    sizes = (
        df[
            [
                "context_id",
                "video_id",
            ]
        ]
        .drop_duplicates()
        .groupby("context_id")
        .size()
    )

    pairs = sizes * (sizes - 1) // 2

    print()
    print("CONTEXT STATISTICS")
    print("------------------")
    print(f"Contexts:              {len(sizes):,}")
    print(f"Mean context size:     {sizes.mean():,.2f}")
    print(f"Median context size:   {sizes.median():,.2f}")
    print(f"P95 context size:      {sizes.quantile(0.95):,.0f}")
    print(f"Max context size:      {sizes.max():,}")
    print(f"Potential pair count:  {pairs.sum():,}")
    print()
