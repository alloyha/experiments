from __future__ import annotations


from collections import defaultdict


from time import perf_counter


import numpy as np


import pandas as pd


from duwhal import Duwhal


from .dataset import sessionize


class DuwhalVideoRecommender:
    """



    YouTube-like two-stage recommender backed directly by Duwhal.







    Candidate sources:







    ItemCF



        - session co-occurrence



        - sparse serving index



        - weighted recent seeds



        - vectorized batch inference







    Graph



        - bounded multi-hop traversal



        - sparse top-K adjacency



        - beam-bounded walk



        - probability scoring



        - optional explanation paths







    Popularity



        - trending popularity



        - exponential time decay







    Hybrid



        - weighted reciprocal-rank fusion



    """

    def __init__(
        self,
        *,
        session_gap_minutes: int = 30,
        max_session_items: int = 15,
        cf_top_k_similar: int = 50,
        cf_min_cooccurrence: int = 2,
        cf_shrinkage: int = 10,
        graph_min_cooccurrence: int = 2,
        graph_top_k_edges: int = 100,
        graph_alpha: float = 0.1,
        graph_max_depth: int = 2,
        graph_beam_width: int = 200,
        popularity_window_days: int = 30,
        popularity_decay_half_life: int = 7,
        memory_limit: str | None = None,
        threads: int | None = None,
    ) -> None:

        self.session_gap_minutes = session_gap_minutes

        self.max_session_items = max_session_items

        self.cf_top_k_similar = cf_top_k_similar

        self.cf_min_cooccurrence = cf_min_cooccurrence

        self.cf_shrinkage = cf_shrinkage

        self.graph_min_cooccurrence = graph_min_cooccurrence

        self.graph_top_k_edges = graph_top_k_edges

        self.graph_alpha = graph_alpha

        self.graph_max_depth = graph_max_depth

        self.graph_beam_width = graph_beam_width

        self.popularity_window_days = popularity_window_days

        self.popularity_decay_half_life = popularity_decay_half_life

        self.memory_limit = memory_limit

        self.threads = threads

        self.train: pd.DataFrame | None = None

        self.contexts: pd.DataFrame | None = None

        self.db: Duwhal | None = None

        self.training_times: dict[str, float] = {}

        # Immutable user-state caches built once during fit().

        # They remove repeated pandas filtering/sorting from serving paths.

        self._history_by_user: dict[int, tuple[int, ...]] = {}

        self._watched_by_user: dict[int, frozenset[int]] = {}

        self._watched_sorted_by_user: dict[int, tuple[int, ...]] = {}

    # ------------------------------------------------------------------

    # Lifecycle

    # ------------------------------------------------------------------

    def _require_fit(self) -> Duwhal:

        if self.db is None:
            raise RuntimeError("Model has not been fitted.")

        return self.db

    def _timed(self, name: str, fn):

        started = perf_counter()

        result = fn()

        elapsed = perf_counter() - started

        self.training_times[name] = elapsed

        print(f"{name}: {elapsed:,.4f}s")

        return result

    def fit(
        self,
        train: pd.DataFrame,
        *,
        components: set[str] | None = None,
    ) -> "DuwhalVideoRecommender":

        if components is None:
            components = {
                "cf",
                "graph",
                "popular",
            }

        self.train = train.copy().reset_index(drop=True)

        self.training_times = {}

        self._timed(
            "User state cache",
            self._build_user_state_cache,
        )

        self.contexts = self._timed(
            "Sessionization",
            lambda: sessionize(
                self.train,
                session_gap_minutes=self.session_gap_minutes,
                max_session_items=self.max_session_items,
            ),
        )

        self.db = Duwhal(
            memory_limit=self.memory_limit,
            threads=self.threads,
        )

        loaded = self._timed(
            "Duwhal ingestion",
            lambda: self.db.load_interactions(
                self.contexts,
                set_col="context_id",
                node_col="video_id",
                sort_col="timestamp",
            ),
        )

        print(
            "Duwhal interaction rows:",
            f"{loaded:,}",
        )

        if "cf" in components:
            self._timed(
                "Duwhal ItemCF fit",
                lambda: self.db.fit_cf(
                    metric="jaccard",
                    min_cooccurrence=self.cf_min_cooccurrence,
                    top_k_similar=self.cf_top_k_similar,
                    shrinkage=self.cf_shrinkage,
                ),
            )

        if "graph" in components:
            self._timed(
                "Duwhal Graph fit",
                lambda: self.db.fit_graph(
                    min_cooccurrence=self.graph_min_cooccurrence,
                    top_k_edges=self.graph_top_k_edges,
                    alpha=self.graph_alpha,
                ),
            )

        if "popular" in components:
            self._timed(
                "Duwhal Popularity fit",
                lambda: self.db.fit_popularity(
                    strategy="trending",
                    window_days=self.popularity_window_days,
                    decay_half_life=self.popularity_decay_half_life,
                ),
            )

        return self

    def close(self) -> None:

        if self.db is not None:
            self.db.close()

            self.db = None

    # ------------------------------------------------------------------

    # Diagnostics

    # ------------------------------------------------------------------

    def print_training_times(self) -> None:

        print()

        print("=" * 70)

        print("DUWHAL FIT TIMES")

        print("=" * 70)

        total = 0.0

        for name, value in self.training_times.items():
            total += value

            print(f"{name:<40}{value:>12,.4f}s")

        print("-" * 70)

        print(f"{'TOTAL':<40}{total:>12,.4f}s")

    def print_model_stats(self) -> None:

        db = self._require_fit()

        table = db.model_stats()

        print()

        print("=" * 70)

        print("DUWHAL MODEL STATS")

        print("=" * 70)

        print()

        if table.num_rows == 0:
            print("No model statistics.")

            return

        print(table.to_pandas().to_string(index=False))

    # ------------------------------------------------------------------

    # User state

    # ------------------------------------------------------------------

    def _build_user_state_cache(
        self,
    ) -> None:
        """



        Precompute immutable user-history and watched-item state.



        This removes repeated pandas filtering and sorting from the

        recommendation hot path.



        """

        if self.train is None:
            raise RuntimeError("Model has not been fitted.")

        if self.train.empty:
            self._history_by_user = {}

            self._watched_by_user = {}

            self._watched_sorted_by_user = {}

            return

        state = self.train[
            [
                "user_id",
                "video_id",
                "timestamp",
            ]
        ].copy()

        state["user_id"] = pd.to_numeric(state["user_id"]).astype(int)

        state["video_id"] = pd.to_numeric(state["video_id"]).astype(int)

        # Stable ordering keeps the newest observation first while

        # preserving deterministic source order for timestamp ties.

        ordered = state.sort_values(
            [
                "user_id",
                "timestamp",
            ],
            ascending=[
                True,
                False,
            ],
            kind="stable",
        )

        unique_history = ordered.drop_duplicates(
            [
                "user_id",
                "video_id",
            ],
            keep="first",
        )

        self._history_by_user = {
            int(user_id): tuple(group["video_id"].astype(int).tolist())
            for user_id, group in unique_history.groupby(
                "user_id",
                sort=False,
            )
        }

        self._watched_by_user = {
            int(user_id): frozenset(group["video_id"].astype(int).tolist())
            for user_id, group in state.groupby(
                "user_id",
                sort=False,
            )
        }

        self._watched_sorted_by_user = {
            user_id: tuple(sorted(items))
            for user_id, items in self._watched_by_user.items()
        }

    def user_history(
        self,
        user_id: int,
        *,
        n: int = 5,
    ) -> list[int]:

        if self.train is None:
            raise RuntimeError("Model has not been fitted.")

        history = self._history_by_user.get(
            int(user_id),
            (),
        )

        return list(history[:n])

    def watched_items(
        self,
        user_id: int,
    ) -> set[int]:

        if self.train is None:
            raise RuntimeError("Model has not been fitted.")

        return set(
            self._watched_by_user.get(
                int(user_id),
                frozenset(),
            )
        )

    @staticmethod
    def seed_weights(
        seeds: list[int],
        *,
        decay: float = 0.85,
    ) -> dict[int, float]:
        """



        Give larger weights to more recent items.







        user_history() is returned newest-first.



        """

        return {item: decay**rank for rank, item in enumerate(seeds)}

    # ------------------------------------------------------------------

    # Arrow -> pandas normalization

    # ------------------------------------------------------------------

    @staticmethod
    def _video_frame(
        table,
    ) -> pd.DataFrame:

        if table.num_rows == 0:
            return pd.DataFrame()

        frame = table.to_pandas()

        if "recommended_item" in frame.columns:
            frame = frame.rename(
                columns={
                    "recommended_item": "video_id",
                }
            )

        elif "item_id" in frame.columns:
            frame = frame.rename(
                columns={
                    "item_id": "video_id",
                }
            )

        frame["video_id"] = pd.to_numeric(frame["video_id"]).astype(int)

        return frame

    def _remove_watched(
        self,
        frame: pd.DataFrame,
        user_id: int,
    ) -> pd.DataFrame:

        if frame.empty:
            return frame

        watched = self._watched_by_user.get(
            int(user_id),
            frozenset(),
        )

        if not watched:
            return frame.copy().reset_index(drop=True)

        return frame[~frame["video_id"].isin(watched)].copy().reset_index(drop=True)

    def _split_batch_table(
        self,
        table,
        users_by_basket: list[int],
        *,
        item_column: str,
        k: int,
        exclude_watched: bool,
        keep_basket_id: bool = False,
    ) -> dict[int, pd.DataFrame]:
        """Convert one Duwhal batch table into per-user DataFrames.

        The hot path performs one Arrow -> pandas conversion, one item-id
        conversion, optional watched filtering across the whole batch, and
        positional slicing by basket boundaries.
        """

        results = {int(user_id): pd.DataFrame() for user_id in users_by_basket}

        if table.num_rows == 0 or not users_by_basket:
            return results

        frame = table.to_pandas()

        if frame.empty:
            return results

        if item_column not in frame.columns:
            raise KeyError(
                f"Batch result does not contain expected item column {item_column!r}."
            )

        if item_column != "video_id":
            frame.rename(
                columns={item_column: "video_id"},
                inplace=True,
            )

        if not pd.api.types.is_integer_dtype(frame["video_id"].dtype):
            frame["video_id"] = frame["video_id"].to_numpy(
                dtype=np.int64,
                copy=False,
            )

        basket_ids = frame["basket_id"].to_numpy(
            dtype=np.int64,
            copy=False,
        )

        if basket_ids.size:
            min_basket = int(basket_ids.min())
            max_basket = int(basket_ids.max())

            if min_basket < 0 or max_basket >= len(users_by_basket):
                raise ValueError(
                    "Batch result contains a basket_id outside the "
                    "submitted basket range."
                )

        # Native Duwhal batch implementations currently group rows by basket.
        # Keep that fast path, but defend against interleaved rows.
        if basket_ids.size > 1 and np.any(basket_ids[1:] < basket_ids[:-1]):
            order = np.argsort(
                basket_ids,
                kind="stable",
            )
            frame = frame.iloc[order]
            basket_ids = basket_ids[order]

        if exclude_watched and basket_ids.size:
            user_lookup = np.asarray(
                users_by_basket,
                dtype=np.int64,
            )
            row_users = user_lookup[basket_ids]
            video_ids = frame["video_id"].to_numpy(
                dtype=np.int64,
                copy=False,
            )
            watched_by_user = self._watched_by_user
            empty_watched = frozenset()

            keep = np.fromiter(
                (
                    int(video_id)
                    not in watched_by_user.get(
                        int(user_id),
                        empty_watched,
                    )
                    for user_id, video_id in zip(row_users, video_ids)
                ),
                dtype=bool,
                count=len(frame),
            )

            if not keep.all():
                frame = frame.loc[keep]
                basket_ids = basket_ids[keep]

        if frame.empty:
            return results

        payload = frame if keep_basket_id else frame.drop(columns=["basket_id"])

        boundaries = np.flatnonzero(
            np.r_[
                True,
                basket_ids[1:] != basket_ids[:-1],
                True,
            ]
        )

        for start, stop in zip(
            boundaries[:-1],
            boundaries[1:],
        ):
            basket_id = int(basket_ids[start])
            user_id = int(users_by_basket[basket_id])
            capped_stop = min(
                int(stop),
                int(start) + k,
            )

            results[user_id] = payload.iloc[int(start) : capped_stop].reset_index(
                drop=True
            )

        return results

    # ------------------------------------------------------------------

    # Individual strategies

    # ------------------------------------------------------------------

    def recommend_cf(
        self,
        user_id: int,
        *,
        k: int = 10,
        history_size: int = 5,
        candidate_pool: int = 250,
    ) -> pd.DataFrame:

        db = self._require_fit()

        seeds = self.user_history(
            user_id,
            n=history_size,
        )

        if not seeds:
            return pd.DataFrame()

        weights = self.seed_weights(seeds)

        table = db.recommend(
            seed_items=seeds,
            strategy="cf",
            n=candidate_pool,
            exclude_seed=True,
            seed_weights=weights,
        )

        frame = self._video_frame(table)

        frame = self._remove_watched(
            frame,
            user_id,
        )

        return frame.head(k).reset_index(drop=True)

    def recommend_graph(
        self,
        user_id: int,
        *,
        k: int = 10,
        history_size: int = 5,
        candidate_pool: int = 200,
        return_paths: bool = False,
    ) -> pd.DataFrame:

        db = self._require_fit()

        seeds = self.user_history(
            user_id,
            n=history_size,
        )

        if not seeds:
            return pd.DataFrame()

        table = db.recommend_graph(
            seeds,
            max_depth=self.graph_max_depth,
            min_weight=1,
            n=candidate_pool,
            exclude_seed=True,
            scoring="probability",
            return_paths=return_paths,
            beam_width=self.graph_beam_width,
        )

        frame = self._video_frame(table)

        frame = self._remove_watched(
            frame,
            user_id,
        )

        return frame.head(k).reset_index(drop=True)

    def recommend_popular(
        self,
        user_id: int,
        *,
        k: int = 10,
    ) -> pd.DataFrame:

        db = self._require_fit()

        watched = self._watched_sorted_by_user.get(
            int(user_id),
            (),
        )

        table = db.recommend(
            strategy="popularity",
            n=k,
            exclude_items=list(watched),
        )

        return self._video_frame(table).head(k).reset_index(drop=True)

    # ------------------------------------------------------------------

    # Hybrid

    # ------------------------------------------------------------------

    def recommend_hybrid(
        self,
        user_id: int,
        *,
        k: int = 10,
        history_size: int = 5,
        candidate_pool: int = 200,
        rrf_constant: int = 60,
        cf_weight: float = 1.00,
        graph_weight: float = 0.80,
        popularity_weight: float = 0.30,
    ) -> pd.DataFrame:

        cf = self.recommend_cf(
            user_id,
            k=candidate_pool,
            history_size=history_size,
            candidate_pool=candidate_pool,
        )

        graph = self.recommend_graph(
            user_id,
            k=candidate_pool,
            history_size=history_size,
            candidate_pool=candidate_pool,
            return_paths=False,
        )

        popular = self.recommend_popular(
            user_id,
            k=min(
                candidate_pool,
                100,
            ),
        )

        scores: dict[int, float] = defaultdict(float)

        cf_rrf: dict[int, float] = defaultdict(float)

        graph_rrf: dict[int, float] = defaultdict(float)

        popular_rrf: dict[int, float] = defaultdict(float)

        source_count: dict[int, int] = defaultdict(int)

        sources = (
            (
                cf,
                cf_weight,
                cf_rrf,
            ),
            (
                graph,
                graph_weight,
                graph_rrf,
            ),
            (
                popular,
                popularity_weight,
                popular_rrf,
            ),
        )

        for (
            frame,
            weight,
            component,
        ) in sources:
            if frame.empty:
                continue

            for rank, row in enumerate(
                frame.itertuples(index=False),
                start=1,
            ):
                video_id = int(row.video_id)

                value = weight / (rrf_constant + rank)

                scores[video_id] += value

                component[video_id] += value

                source_count[video_id] += 1

        if not scores:
            return pd.DataFrame()

        rows = [
            {
                "video_id": video_id,
                "score": score,
                "cf_rrf": cf_rrf[video_id],
                "graph_rrf": graph_rrf[video_id],
                "popular_rrf": popular_rrf[video_id],
                "retrieval_sources": source_count[video_id],
            }
            for video_id, score in scores.items()
        ]

        return (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "score",
                    "retrieval_sources",
                    "video_id",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .head(k)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------

    # Vectorized CF batch

    # ------------------------------------------------------------------

    def recommend_cf_batch(
        self,
        user_ids: list[int],
        *,
        k: int = 10,
        history_size: int = 5,
        candidate_pool: int = 250,
    ) -> tuple[
        dict[int, pd.DataFrame],
        float,
    ]:
        db = self._require_fit()

        baskets: list[dict[int, float]] = []
        valid_users: list[int] = []

        for raw_user_id in user_ids:
            user_id = int(raw_user_id)
            seeds = self._history_by_user.get(
                user_id,
                (),
            )[:history_size]

            if not seeds:
                continue

            baskets.append(self.seed_weights(list(seeds)))
            valid_users.append(user_id)

        if not baskets:
            return {}, 0.0

        started = perf_counter()

        table = db.recommend_batch(
            baskets,
            strategy="cf",
            n=candidate_pool,
            exclude_seed=True,
        )

        elapsed = perf_counter() - started

        return (
            self._split_batch_table(
                table,
                valid_users,
                item_column="item_id",
                k=k,
                exclude_watched=True,
                # Preserve the existing CF per-user frame schema.
                keep_basket_id=True,
            ),
            elapsed,
        )

    # ------------------------------------------------------------------

    # Graph return_paths diagnostics

    # ------------------------------------------------------------------

    def benchmark_graph_paths(
        self,
        user_id: int,
        *,
        history_size: int = 5,
        n: int = 20,
        repeats: int = 5,
        rtol: float = 1e-12,
        atol: float = 1e-15,
    ) -> dict[str, object]:
        """



        Compare Graph execution with and without explanation paths.







        This deliberately separates four notions:







        same_items



            Same recommendation set.







        same_ranking



            Same recommendation order.







        same_hops



            Same min_hops for each recommended item.







        scores_close



            Scores are numerically equivalent under np.allclose().







        `same_semantics` means the candidate set, hop counts and scores



        are equivalent. Ranking is reported separately because numerically



        tied scores may legitimately swap order unless the library enforces



        an explicit deterministic final tie-break.



        """

        db = self._require_fit()

        seeds = self.user_history(
            user_id,
            n=history_size,
        )

        if not seeds:
            raise ValueError(f"User {user_id} has no training history.")

        pathless_times: list[float] = []

        paths_times: list[float] = []

        fast = None

        explained = None

        repeats = max(
            int(repeats),
            1,
        )

        # Both variants are already expected to be warm when this method

        # is normally called from main.py. Repeating reduces scheduler noise.

        for _ in range(repeats):
            started = perf_counter()

            fast = db.recommend_graph(
                seeds,
                max_depth=self.graph_max_depth,
                min_weight=1,
                n=n,
                exclude_seed=True,
                scoring="probability",
                return_paths=False,
                beam_width=self.graph_beam_width,
            )

            pathless_times.append(perf_counter() - started)

            started = perf_counter()

            explained = db.recommend_graph(
                seeds,
                max_depth=self.graph_max_depth,
                min_weight=1,
                n=n,
                exclude_seed=True,
                scoring="probability",
                return_paths=True,
                beam_width=self.graph_beam_width,
            )

            paths_times.append(perf_counter() - started)

        assert fast is not None

        assert explained is not None

        fast_df = self._video_frame(fast).reset_index(drop=True)

        explained_df = self._video_frame(explained).reset_index(drop=True)

        # --------------------------------------------------------------

        # Ranking-level comparison

        # --------------------------------------------------------------

        fast_ranked_items = fast_df["video_id"].astype(int).tolist()

        explained_ranked_items = explained_df["video_id"].astype(int).tolist()

        same_ranking = fast_ranked_items == explained_ranked_items

        fast_item_set = set(fast_ranked_items)

        explained_item_set = set(explained_ranked_items)

        same_items = fast_item_set == explained_item_set

        missing_from_paths = sorted(fast_item_set - explained_item_set)

        missing_from_pathless = sorted(explained_item_set - fast_item_set)

        # --------------------------------------------------------------

        # Align by item ID before comparing numerical/model semantics.

        # --------------------------------------------------------------

        fast_cmp = fast_df[
            [
                "video_id",
                "total_strength",
                "min_hops",
            ]
        ].rename(
            columns={
                "total_strength": "pathless_score",
                "min_hops": "pathless_hops",
            }
        )

        explained_cmp = explained_df[
            [
                "video_id",
                "total_strength",
                "min_hops",
            ]
        ].rename(
            columns={
                "total_strength": "paths_score",
                "min_hops": "paths_hops",
            }
        )

        comparison = (
            fast_cmp.merge(
                explained_cmp,
                on="video_id",
                how="outer",
                indicator=True,
            )
            .sort_values("video_id")
            .reset_index(drop=True)
        )

        both = comparison[comparison["_merge"] == "both"].copy()

        same_hops = (
            same_items
            and (
                both["pathless_hops"].astype(int).to_numpy()
                == both["paths_hops"].astype(int).to_numpy()
            ).all()
        )

        if same_items and len(both) == len(fast_df) == len(explained_df):
            pathless_scores = both["pathless_score"].to_numpy(dtype=float)

            paths_scores = both["paths_score"].to_numpy(dtype=float)

            score_deltas = np.abs(pathless_scores - paths_scores)

            scores_close = bool(
                np.allclose(
                    pathless_scores,
                    paths_scores,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=False,
                )
            )

            max_score_delta = float(score_deltas.max()) if len(score_deltas) else 0.0

            mean_score_delta = float(score_deltas.mean()) if len(score_deltas) else 0.0

        else:
            scores_close = False

            max_score_delta = None

            mean_score_delta = None

        same_semantics = bool(same_items and same_hops and scores_close)

        return {
            "pathless_seconds": float(np.median(pathless_times)),
            "paths_seconds": float(np.median(paths_times)),
            "pathless_times": pathless_times,
            "paths_times": paths_times,
            "same_items": same_items,
            "same_ranking": same_ranking,
            "same_hops": bool(same_hops),
            "scores_close": scores_close,
            "same_semantics": same_semantics,
            "max_score_delta": max_score_delta,
            "mean_score_delta": mean_score_delta,
            "missing_from_paths": missing_from_paths,
            "missing_from_pathless": missing_from_pathless,
            "pathless": fast_df,
            "explained": explained_df,
            "comparison": comparison,
        }

    def recommend_popular_batch(
        self,
        user_ids: list[int],
        *,
        k: int = 10,
    ) -> tuple[
        dict[int, pd.DataFrame],
        float,
    ]:
        db = self._require_fit()

        normalized_users = [int(user_id) for user_id in user_ids]

        # Reuse the immutable cached tuples rather than rebuilding 100 lists.
        exclusions = [
            self._watched_sorted_by_user.get(
                user_id,
                (),
            )
            for user_id in normalized_users
        ]

        started = perf_counter()

        table = db.recommend_popular_batch(
            exclusions,
            n=k,
        )

        batch_seconds = perf_counter() - started

        return (
            self._split_batch_table(
                table,
                normalized_users,
                item_column="item_id",
                k=k,
                exclude_watched=False,
                keep_basket_id=False,
            ),
            batch_seconds,
        )

    def recommend_graph_batch(
        self,
        user_ids: list[int],
        *,
        k: int = 10,
        history_size: int = 5,
        candidate_pool: int = 200,
        return_paths: bool = False,
    ) -> tuple[
        dict[int, pd.DataFrame],
        float,
    ]:
        db = self._require_fit()

        baskets: list[tuple[int, ...]] = []
        valid_users: list[int] = []

        for raw_user_id in user_ids:
            user_id = int(raw_user_id)
            seeds = self._history_by_user.get(
                user_id,
                (),
            )[:history_size]

            if not seeds:
                continue

            baskets.append(seeds)
            valid_users.append(user_id)

        if not baskets:
            return {}, 0.0

        started = perf_counter()

        table = db.recommend_batch(
            baskets,
            strategy="graph",
            n=candidate_pool,
            max_depth=self.graph_max_depth,
            min_weight=1,
            exclude_seed=True,
            scoring="probability",
            return_paths=return_paths,
            beam_width=self.graph_beam_width,
        )

        elapsed = perf_counter() - started

        return (
            self._split_batch_table(
                table,
                valid_users,
                item_column="recommended_item",
                k=k,
                exclude_watched=True,
                keep_basket_id=False,
            ),
            elapsed,
        )
