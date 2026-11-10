"""
clean_phase3.py
Cleans PHASE_3_CLEANED_DATASETS.xlsx (wide, one row per area) into a Power BI / Excel-ready table.

    python clean_phase3.py --input PHASE_3_CLEANED_DATASETS.xlsx --output PHASE_3_FINAL_CLEAN.xlsx
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = []


def log(step, detail):
    LOG.append({"step": step, "detail": detail})


# Explicit renames: drop doubled prefixes, standardise units, snake_case
RENAME = {
    "Area": "area", "Record Timestamp": "record_timestamp",
    "Latitude": "latitude", "Longitude": "longitude",
    "River_River_Name": "river_name", "River_Station": "river_station_id",
    "Weather_Wind_Speed_kmph": "weather_wind_speed_kmh",
    "Road_Road_ID": "road_id", "Shelter_Available_Beds": "shelter_available_beds",
    "Feature_Shelter_Occupancy_pct": "feature_shelter_occupancy_pct",
    "Feature_Shelter_Available_pct": "feature_shelter_available_pct",
    "Feature_Data_Completeness_pct": "feature_data_completeness_pct",
}


def snake(c):
    return re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_").lower()


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(here / "PHASE_3_CLEANED_DATASETS.xlsx"))
    ap.add_argument("--output", default=str(here / "PHASE_3_FINAL_CLEAN.xlsx"))
    a = ap.parse_args()

    # 1. Header is on row 3 (rows 1-2 are merged title/note rows)
    df = pd.read_excel(a.input, sheet_name=0, header=2)
    log("remove title rows", "dropped merged title/note rows 1-2; header promoted to row 1")
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # 2. Column names
    before = list(df.columns)
    df.columns = [RENAME.get(c, snake(c)) for c in df.columns]
    log("rename columns", f"{sum(b != n for b, n in zip(before, df.columns))} of {len(df.columns)} renamed to "
        "snake_case (doubled prefixes removed, kmph -> kmh)")

    # 3. Text trim + duplicates
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    n = len(df)
    df = df.drop_duplicates(subset=["area"]).reset_index(drop=True)
    log("duplicates", f"{n - len(df)} duplicate area rows removed")

    # 4. Types
    for c in ["record_timestamp", "rainfall_date", "historical_last_flood_date",
              "satellite_image_date", "alert_time"]:
        df[c] = pd.to_datetime(df[c])
    df["rescue_team_availability"] = df["rescue_team_availability"].astype(bool)
    log("types", "date columns -> datetime, availability -> boolean")

    # 5. Integrity checks on source values
    nulls = int(df.isna().sum().sum())
    log("missing values", f"{nulls} nulls found")
    assert (df["shelter_occupied"] <= df["shelter_capacity"]).all()
    assert (df["prediction_risk_score"].between(0, 100)).all()
    log("range checks", "occupied <= capacity, risk score in 0-100, all passed")

    # 6. Recompute feature columns from source columns (no trust in cached values)
    df["shelter_available_beds"] = df["shelter_capacity"] - df["shelter_occupied"]
    df["feature_river_level_gap_m"] = (df["river_water_level_m"] - df["river_danger_level_m"]).round(2)
    df["feature_shelter_occupancy_pct"] = (df["shelter_occupied"] / df["shelter_capacity"] * 100).round(1)
    df["feature_shelter_available_pct"] = (100 - df["feature_shelter_occupancy_pct"]).round(1)
    df["feature_risk_band_code"] = df["prediction_risk"].map({"High": 3, "Medium": 2, "Low": 1}).astype(int)
    df["feature_arrival_urgency"] = np.select(
        [df["prediction_arrival_time_hr"] <= 3, df["prediction_arrival_time_hr"] <= 8],
        ["Immediate", "Urgent"], "Monitor")
    df["feature_evacuation_priority"] = np.select(
        [(df["prediction_risk"] == "High") & (df["prediction_arrival_time_hr"] <= 3),
         df["prediction_risk"] == "High", df["prediction_risk"] == "Medium"],
        ["Critical", "High", "Medium"], "Routine")
    src_cols = [c for c in df.columns if not c.startswith("feature_")]
    df["feature_data_completeness_pct"] = (df[src_cols].notna().mean(axis=1) * 100).round(1)
    log("feature columns", "recomputed from source columns; float noise removed (e.g. 0.3799999 -> 0.38); "
        "pct columns now stored as 0-100 numbers instead of 0-1 fractions")

    # 7. Contradiction fix: a team that is Deployed is not available
    avail = df["rescue_team_status"].isin(["Available", "On standby"])
    changed = df.loc[df["rescue_team_availability"] != avail, "area"].tolist()
    df["rescue_team_availability"] = avail
    log("consistency fix", f"rescue_team_availability set from status (Available/On standby = True); changed: {changed}")

    # 8. Flags (reported, not altered)
    full = df.loc[df["population_at_risk"] >= df["population_total"], "area"].tolist()
    log("flag", f"population_at_risk equals total population (100%) for {full} - verify with source")
    log("flag", f"alert_status values: {sorted(df['alert_status'].unique())} - mixed wording, left unchanged")
    log("flag", "river_station_id uses area initials (e.g. RT-02) and can look like rescue_team_id RT-02; "
        "do not join on these columns")
    log("summary", f"{df.shape[0]} rows x {df.shape[1]} columns")

    # 9. Data dictionary
    groups = {"rainfall": "Rainfall", "river": "River gauge", "weather": "Weather forecast",
              "historical": "Past floods", "satellite": "Satellite", "population": "Population",
              "rescue": "Rescue teams", "shelter": "Shelter", "road": "Roads", "alert": "Alerts",
              "prediction": "Flood prediction", "feature": "Derived feature"}
    dd = pd.DataFrame({"column": df.columns,
                       "group": [next((v for k, v in groups.items() if c.startswith(k)), "Core") for c in df.columns],
                       "dtype": [str(t) for t in df.dtypes]})

    # 10. Write
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo
    with pd.ExcelWriter(a.output, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm",
                        date_format="yyyy-mm-dd") as xw:
        for name, d in [("clean_dataset", df), ("data_dictionary", dd), ("cleaning_log", pd.DataFrame(LOG))]:
            d.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "B2" if name == "clean_dataset" else "A2"
            for i, col in enumerate(d.columns, 1):
                w = max(len(str(col)), *(len(str(v)) for v in d[col])) + 2
                ws.column_dimensions[get_column_letter(i)].width = min(w, 60)
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            t = Table(displayName=name.title().replace("_", ""), ref=ws.dimensions)
            t.tableStyleInfo = TableStyleInfo(name="TableStyleLight9", showRowStripes=True)
            ws.add_table(t)
    print("written", a.output)
    print(pd.DataFrame(LOG).to_string(index=False))


if __name__ == "__main__":
    main()
