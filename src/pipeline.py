from pathlib import Path
import polars as pl
from src.features import (
    apply_baseline_shrinkage,
    apply_qb_reversion,
    attach_current_team,
    build_veteran_feature_table,
)
from src.situational import build_situational_features
from src.rookies import build_rookie_feature_table
from src.adp import attach_adp
from src.ranking import apply_free_agents, apply_situational_weights

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "player_features.csv"
SEASONS = [2023, 2024, 2025]


def build_player_feature_table(output_path=DEFAULT_OUTPUT_PATH):
    veteran_features = build_veteran_feature_table(SEASONS)
    veteran_features = attach_current_team(veteran_features)
    veteran_features = veteran_features.with_columns([
        pl.lit(False).alias("is_rookie"),
        pl.lit(False).alias("baseline_low_confidence"),
    ])

    rookie_features = build_rookie_feature_table()

    combined = pl.concat([veteran_features, rookie_features], how="diagonal")

    situational_features = build_situational_features(SEASONS, combined)

    full_table = combined.join(
        situational_features.drop(["team", "position"]),
        on="player_id",
        how="left",
    )

    # Phase 11 B (CP5). Shrink before the situational weights are applied
    # -- they were fitted against shrunk deltas in the backtest, so they
    # must be applied on top of a shrunk baseline here or the two halves
    # of the model disagree.
    #
    # Rookies are excluded. `games_played` for a rookie describes a
    # college career the model never saw, and their baseline is a cohort
    # projection, not personal history. Phase 12 handles their confidence
    # separately.
    full_table = apply_baseline_shrinkage(
        full_table, exclude=pl.col("is_rookie")
    )

    # Phase 15b. AFTER shrinkage, because shrinkage leaves quarterbacks
    # untouched (QB is in SHRINKAGE_EXCLUDED_POSITIONS) and this
    # overwrites the columns it left at their pass-through values. Before
    # the weights, for the same reason shrinkage is: QB carries no
    # situational weights today, but if it ever does they must sit on top
    # of the baseline they were fitted against.
    #
    # No-op when data/qb_reversion.json is absent, which reproduces v17.
    full_table = apply_qb_reversion(full_table)

    full_table = attach_adp(full_table)

    # BEFORE the weights are applied. A free agent's team-derived
    # features have to be blanked while they can still change the
    # adjustment -- doing it downstream in build_board would edit a
    # column that nothing reads again.
    full_table = apply_free_agents(full_table)

    full_table = apply_situational_weights(full_table)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_table.write_csv(output_path)
    print(f"Wrote {full_table.shape[0]} players to {output_path}")


if __name__ == "__main__":
    build_player_feature_table()