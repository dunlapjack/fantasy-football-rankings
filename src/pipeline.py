from pathlib import Path
from src.features import build_veteran_feature_table, attach_current_team
from src.situational import build_situational_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "player_features.csv"
SEASONS = [2023, 2024, 2025]


def build_player_feature_table(output_path=DEFAULT_OUTPUT_PATH):
    veteran_features = build_veteran_feature_table(SEASONS)
    veteran_features = attach_current_team(veteran_features)

    situational_features = build_situational_features(SEASONS, veteran_features)

    full_table = veteran_features.join(
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