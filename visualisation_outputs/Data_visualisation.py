"""
================================================================================
SR UNIVERSITY - DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING (CYBER SECURITY)
COURSE: DATA ANALYTICS AND VISUALIZATION PROJECT (CAPSTONE)
PROJECT: FLOOD ALERT & RESCUE COORDINATION SYSTEM
MODULE: HIGH-RESOLUTION PUBLICATION-GRADE DATA VISUALIZATION ENGINE
AUTHORS:
  - S. Sai Sathwik (Roll No: 2503A53021) - Project Proposal & Interactive Dashboard
  - P. Hruthik Reddy (Roll No: 2503A53029) - Data Collection & Visualization
  - G. Srikar (Roll No: 2503A53028) - Data Cleaning & Analytics
================================================================================
"""

import os
import sys

# Ensure Matplotlib config directory is safely set in writable temp directory
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "."), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless server rendering
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Define aesthetic academic color palette
NAVY_PRIMARY = "#1E293B"
ACCENT_BLUE = "#2563EB"
CRIMSON_DANGER = "#EF4444"
AMBER_WARNING = "#F59E0B"
EMERALD_SUCCESS = "#10B981"
SLATE_MUTED = "#64748B"
LIGHT_BG = "#F8FAFC"

def set_academic_style():
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E2E8F0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

def generate_all_visualisations():
    data_path = os.path.join("clean_dataset", "clean_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Clean dataset not found at {data_path}. Please run data_cleaning.py first.")

    out_dir = "visualisations"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[INFO] Ingesting dataset for visualization from {data_path}...")
    df = pd.read_csv(data_path)
    set_academic_style()

    # -------------------------------------------------------------------------
    # FIG 1: Primary Target Distribution with Dual KDE Curve (Risk Score & Rainfall)
    # -------------------------------------------------------------------------
    print("  [+] Generating Fig 1: Primary Distribution Overview (300 DPI)...")
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Histogram of Prediction Risk Score
    sns.histplot(
        df["Prediction_Risk_Score"], 
        kde=True, 
        color=ACCENT_BLUE, 
        bins=35, 
        ax=ax1, 
        stat="density",
        alpha=0.45,
        edgecolor=NAVY_PRIMARY,
        linewidth=0.7,
        label="Flood Risk Score (Empirical PDF)"
    )
    
    # Annotate Key Thresholds
    ax1.axvline(35, color=EMERALD_SUCCESS, linestyle="--", linewidth=1.8, label="Moderate Threshold (35)")
    ax1.axvline(60, color=AMBER_WARNING, linestyle="--", linewidth=1.8, label="High Risk Threshold (60)")
    ax1.axvline(80, color=CRIMSON_DANGER, linestyle="--", linewidth=2.0, label="Critical Evacuation Threshold (80)")
    
    ax1.set_title("Figure 1: Empirical Distribution of Multi-Factor Flood Risk Score", fontsize=14, fontweight="bold", pad=15, color=NAVY_PRIMARY)
    ax1.set_xlabel("Composite Flood Risk Score (0 - 100 Scale)", fontsize=11, fontweight="semibold", labelpad=10)
    ax1.set_ylabel("Probability Density Function (PDF)", fontsize=11, fontweight="semibold", labelpad=10)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    
    fig1_path = os.path.join(out_dir, "fig1_distribution_overview.png")
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"      Saved: {fig1_path}")

    # -------------------------------------------------------------------------
    # FIG 2: Annotated Feature Correlation Matrix Heatmap
    # -------------------------------------------------------------------------
    print("  [+] Generating Fig 2: Feature Correlation Heatmap (300 DPI)...")
    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
    
    corr_features = [
        "Rainfall_24h_mm", "Rainfall_Intensity_mm_hr", "River_Water_Level_m",
        "Feature_River_Level_Gap_m", "Satellite_Flood_Depth_m", "Satellite_Flooded_Area_sq_km",
        "Population_At_Risk", "Rescue_Teams_Active", "Feature_Shelter_Occupancy_pct",
        "Prediction_Arrival_Time_hr", "Prediction_Risk_Score"
    ]
    
    feature_labels = [
        "24h Rainfall", "Rain Intensity", "River Level", 
        "River Danger Gap", "Flood Depth", "Flooded Area",
        "Pop At Risk", "Rescue Teams", "Shelter Occupancy %",
        "Arrival Time (hrs)", "Risk Score"
    ]
    
    corr_matrix = df[corr_features].corr(method="pearson")
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap="coolwarm", 
        vmax=1.0, 
        vmin=-1.0, 
        center=0,
        annot=True, 
        fmt=".2f", 
        square=True, 
        linewidths=0.75, 
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        ax=ax
    )
    
    ax.set_title("Figure 2: Multivariate Correlation Matrix of Hydrological & Vulnerability Variables", fontsize=13, fontweight="bold", pad=15, color=NAVY_PRIMARY)
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    
    fig2_path = os.path.join(out_dir, "fig2_correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"      Saved: {fig2_path}")

    # -------------------------------------------------------------------------
    # FIG 3: Key Segment Categorical Comparison (River Water Level vs Danger Mark)
    # -------------------------------------------------------------------------
    print("  [+] Generating Fig 3: Categorical Segment Comparison (300 DPI)...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    area_summary = df.groupby("Area").agg({
        "River_Water_Level_m": "mean",
        "River_Danger_Level_m": "first",
        "Satellite_Flood_Depth_m": "mean"
    }).reset_index()
    
    x = np.arange(len(area_summary["Area"]))
    bar_width = 0.35
    
    rects1 = ax.bar(x - bar_width/2, area_summary["River_Water_Level_m"], bar_width, label="Observed Mean Water Level (m)", color=ACCENT_BLUE, edgecolor=NAVY_PRIMARY, alpha=0.85)
    rects2 = ax.bar(x + bar_width/2, area_summary["River_Danger_Level_m"], bar_width, label="Station Danger Threshold (m)", color=CRIMSON_DANGER, edgecolor=NAVY_PRIMARY, alpha=0.85)
    
    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}m", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}m", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Figure 3: Mean Hydraulic River Water Level vs Station Danger Threshold Across Sectors", fontsize=13, fontweight="bold", pad=15, color=NAVY_PRIMARY)
    ax.set_ylabel("Water Level (Meters)", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(area_summary["Area"], fontsize=10.5, fontweight="semibold")
    ax.set_ylim(0, 10.5)
    ax.legend(loc="upper right", frameon=True)
    
    fig3_path = os.path.join(out_dir, "fig3_category_comparison.png")
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"      Saved: {fig3_path}")

    # -------------------------------------------------------------------------
    # FIG 4: Multi-Variable Scatter Plot with Regression Trend (Rainfall vs Flood Depth)
    # -------------------------------------------------------------------------
    print("  [+] Generating Fig 4: Multi-Variable Scatter & Regression Plot (300 DPI)...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Sample down to 800 points for crisp visual readability in scatter
    sample_df = df.sample(n=min(900, len(df)), random_state=42)
    
    severity_order = ["Low", "Medium", "High", "Critical"]
    palette_dict = {
        "Low": EMERALD_SUCCESS,
        "Medium": AMBER_WARNING,
        "High": "#EA580C",
        "Critical": CRIMSON_DANGER
    }
    
    sns.scatterplot(
        data=sample_df,
        x="Rainfall_24h_mm",
        y="Satellite_Flood_Depth_m",
        hue="Alert_Severity",
        hue_order=severity_order,
        palette=palette_dict,
        alpha=0.75,
        s=60,
        edgecolor="white",
        linewidth=0.5,
        ax=ax
    )
    
    # Add robust linear trendline
    sns.regplot(
        data=df,
        x="Rainfall_24h_mm",
        y="Satellite_Flood_Depth_m",
        scatter=False,
        ax=ax,
        color=NAVY_PRIMARY,
        line_kws={"linewidth": 2.2, "linestyle": "-.", "label": "Hydraulic Trendline (OLS Fit)"}
    )
    
    ax.set_title("Figure 4: Non-Linear Hydrodynamic Relationship: 24h Cumulative Rainfall vs Flood Inundation Depth", fontsize=13, fontweight="bold", pad=15, color=NAVY_PRIMARY)
    ax.set_xlabel("24-Hour Cumulative Rainfall (mm)", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Satellite Observed Inundation Depth (m)", fontsize=11, fontweight="semibold", labelpad=10)
    ax.legend(title="Alert Severity Level", frameon=True, loc="upper left")
    
    fig4_path = os.path.join(out_dir, "fig4_scatter_relationship.png")
    plt.tight_layout()
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"      Saved: {fig4_path}")

    # -------------------------------------------------------------------------
    # FIG 5: Proportion & Composition Breakdown (Donut Chart: Population at Risk & Alert Severity)
    # -------------------------------------------------------------------------
    print("  [+] Generating Fig 5: Composition Breakdown Donut Chart (300 DPI)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)
    
    # Donut 1: Population Vulnerability Breakdown
    total_pop = df["Population_Total"].iloc[0]
    vuln_pop = df["Population_Vulnerable"].mean()
    risk_pop = df["Population_At_Risk"].mean()
    safe_pop = max(0, total_pop - risk_pop)
    
    pop_sizes = [safe_pop, risk_pop, vuln_pop]
    pop_labels = ["Safe / Buffer Pop", "Population At Risk", "Vulnerable (Elderly/Children)"]
    pop_colors = [EMERALD_SUCCESS, AMBER_WARNING, CRIMSON_DANGER]
    
    wedges, texts, autotexts = ax1.pie(
        pop_sizes, 
        labels=pop_labels, 
        autopct="%1.1f%%", 
        startangle=140, 
        colors=pop_colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax1.set_title("A: Demographic Impact Breakdown", fontsize=12, fontweight="bold", color=NAVY_PRIMARY)

    # Donut 2: Incident Severity Distribution
    sev_counts = df["Alert_Severity"].value_counts().reindex(severity_order)
    sev_colors = [EMERALD_SUCCESS, AMBER_WARNING, "#EA580C", CRIMSON_DANGER]
    
    wedges2, texts2, autotexts2 = ax2.pie(
        sev_counts.values,
        labels=sev_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sev_colors,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for at in autotexts2:
        at.set_color("white")
        at.set_fontweight("bold")
    ax2.set_title("B: Historical Alert Severity Distribution", fontsize=12, fontweight="bold", color=NAVY_PRIMARY)

    fig.suptitle("Figure 5: Inundation Demographics & Operational Emergency Alert Composition", fontsize=14, fontweight="bold", y=0.98, color=NAVY_PRIMARY)
    
    fig5_path = os.path.join(out_dir, "fig5_composition_breakdown.png")
    plt.tight_layout()
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"      Saved: {fig5_path}")

    # -------------------------------------------------------------------------
    # FIG 6: Composite Risk & Rescue Response Priority Ranking Bar Chart
    # -------------------------------------------------------------------------
    print("  [+] Generating Fig 6: Composite Index Ranking Chart (300 DPI)...")
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    
    # Calculate composite metrics per area
    ranking_df = df.groupby("Area").agg({
        "Prediction_Risk_Score": "mean",
        "Feature_Shelter_Occupancy_pct": "mean",
        "Rescue_Teams_Active": "mean",
        "Population_At_Risk": "mean"
    }).reset_index()
    
    # Custom Composite Vulnerability & Action Score
    ranking_df["Composite_Action_Score"] = (
        ranking_df["Prediction_Risk_Score"] * 0.45 + 
        ranking_df["Feature_Shelter_Occupancy_pct"] * 0.35 +
        (ranking_df["Population_At_Risk"] / 1000.0) * 0.20
    ).round(2)
    
    ranking_df = ranking_df.sort_values(by="Composite_Action_Score", ascending=True)
    
    y_pos = np.arange(len(ranking_df["Area"]))
    colors = [EMERALD_SUCCESS, "#3B82F6", AMBER_WARNING, "#EA580C", CRIMSON_DANGER]
    
    bars = ax.barh(y_pos, ranking_df["Composite_Action_Score"], color=colors, edgecolor=NAVY_PRIMARY, height=0.6, alpha=0.9)
    
    # Annotate bar values
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"Score: {w:.1f}", 
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0), 
            textcoords="offset points", 
            ha="left", 
            va="center", 
            fontsize=9.5, 
            fontweight="bold"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ranking_df["Area"], fontsize=11, fontweight="semibold")
    ax.set_xlabel("Composite Rescue Priority & Disaster Stress Score (0 - 100)", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_title("Figure 6: Municipal Catchment Ranking by Composite Flood Vulnerability & Action Priority", fontsize=13, fontweight="bold", pad=15, color=NAVY_PRIMARY)
    ax.set_xlim(0, max(ranking_df["Composite_Action_Score"]) * 1.18)
    
    fig6_path = os.path.join(out_dir, "fig6_composite_index_ranking.png")
    plt.tight_layout()
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"      Saved: {fig6_path}")

    print("=" * 80)
    print("[SUCCESS] All 6 high-resolution publication charts generated successfully at 300 DPI!")
    print("=" * 80)

if __name__ == "__main__":
    generate_all_visualisations()
