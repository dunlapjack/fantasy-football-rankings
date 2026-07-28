from pathlib import Path
import polars as pl
from src.features import build_veteran_feature_table, attach_current_team
from src.situational import build_situational_features
from src.rookies import build_rookie_feature_table

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_table.write_csv(output_path)
    print(f"Wrote {full_table.shape[0]} players to {output_path}")


if __name__ == "__main__":
    build_player_feature_table()