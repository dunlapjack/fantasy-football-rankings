import nflreadpy as nfl

stats = nfl.load_player_stats([2025])
print("Rows:", stats.shape[0])
print("Columns:", stats.columns[:10])