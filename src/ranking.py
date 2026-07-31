import polars as pl

# Weights derived from the Phase 5 historical backtest (src/backtest.py,
# 2023-2025 seasons, players with 8+ games played). RB was the only
# position with a statistically significant, individually-clean effect:
# continuity_score (QB change + coach change, 0/1/2) coef=-0.9875,
# p=0.005, n=239. QB/WR/TE showed no combinable signal across every
# specification tested and get no weight -- their
# adjusted_fantasy_points_per_game is identical to the raw baseline.
SITUATIONAL_WEIGHTS = {
    "RB": {"continuity_score": -0.9493, "workload_share": -3.4441, "experience": -0.4674},
    "WR": {"team_changed": -0.8667, "workload_share": -7.1708, "recent_major_injury": -1.8642, "experience": -0.3873},
    "TE": {"workload_share": -7.5826, "experience": -0.1898},
}


def apply_situational_weights(player_features):
    """
    Adds `adjusted_fantasy_points_per_game` on top of the pure
    statistical `fantasy_points_per_game` baseline, applying
    position-specific weights from SITUATIONAL_WEIGHTS. Positions not
    listed in SITUATIONAL_WEIGHTS (currently QB, WR, TE, K, DST) pass
    through with zero adjustment.
    """
    adjustment = pl.lit(0.0)
    for position, weights in SITUATIONAL_WEIGHTS.items():
        position_adjustment = pl.lit(0.0)
        for feature, weight in weights.items():
            position_adjustment = position_adjustment + pl.col(feature).fill_null(0) * weight
        adjustment = pl.when(pl.col("position") == position).then(position_adjustment).otherwise(adjustment)

    return player_features.with_columns(
        (pl.col("fantasy_points_per_game") + adjustment).alias("adjusted_fantasy_points_per_game")
    )