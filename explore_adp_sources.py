import requests
import json

print("=== Fantasy Football Calculator ADP ===")
ffc_resp = requests.get(
    "https://fantasyfootballcalculator.com/api/v1/adp/ppr",
    params={"teams": 12, "year": 2026, "position": "all"},
    timeout=15,
)
print("Status:", ffc_resp.status_code)
ffc_data = ffc_resp.json()
print("Top-level keys:", list(ffc_data.keys()))
print("First player entry:")
print(json.dumps(ffc_data["players"][0], indent=2))
print(f"Total players returned: {len(ffc_data['players'])}")

print("\n=== Sleeper Projections (checking for adp field) ===")
sleeper_resp = requests.get(
    "https://api.sleeper.app/projections/nfl/2026/0",
    params=[
        ("season_type", "regular"),
        ("position[]", "QB"),
        ("position[]", "RB"),
        ("position[]", "WR"),
        ("position[]", "TE"),
    ],
    timeout=15,
)
print("Status:", sleeper_resp.status_code)
sleeper_data = sleeper_resp.json()
print("Response type:", type(sleeper_data))
if isinstance(sleeper_data, list):
    print(f"Total entries: {len(sleeper_data)}")
    print("First entry:")
    print(json.dumps(sleeper_data[0], indent=2))
else:
    print(json.dumps(sleeper_data, indent=2)[:2000])

print("\n=== Sleeper ADP field sanity check ===")
gibbs_entries = [
    e for e in sleeper_data
    if e["player"]["last_name"] == "Gibbs" and e["player"]["first_name"] == "Jahmyr"
]
print(f"Found {len(gibbs_entries)} entries for Jahmyr Gibbs")
for e in gibbs_entries:
    print(json.dumps(e, indent=2))

adp_values = [e["stats"].get("adp_dd_ppr") for e in sleeper_data if "adp_dd_ppr" in e["stats"]]
print(f"\nEntries with adp_dd_ppr present: {len(adp_values)} / {len(sleeper_data)}")
print(f"Min: {min(adp_values)}, Max: {max(adp_values)}")
print(f"Number exactly equal to 1000.0: {sum(1 for v in adp_values if v == 1000.0)}")