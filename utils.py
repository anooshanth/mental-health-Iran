import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency
from pywaffle import Waffle
import itertools


illness_related_cols = [
    "illness-performance",
    "illness-discrimination",
    "illness-selfharm-thought",
    "illness-selfharm-action",
    "illness-personal-relationships",
    "illness-humiliation",
]

treatment_related_cols = [
    "treatment-abondonment-expenses",
    "treatment-abondonment-medication-side-effects",
    "treatment-abondonment-distrust",
    "treatment-abondonment-lackhope",
    "treatment-abondonment-lackknowledge",
    "treatment-abondonment-resources",
    "treatment-abondonment-support-network",
    "treatment-abondonment-stigma",
    "treatment-abondonment-unprofessional-behavior",
]

event_related_cols = [
    "event-workshops",
    "event-dr-visits",
    "event-grouptherapy",
    "event-illness-workshops",
    "event-live-QA",
    "event-friendly-gatherings",
]

illness_stigma_cols = [
    "illness-humiliation",
    "illness-discrimination",
    "illness-personal-relationships",
]

illness_personal_cols = [
    "illness-performance",
    "illness-selfharm-thought",
    "illness-selfharm-action",
]

illness_impact_map = {
    "illness-performance": "Work Performance Issues",
    "illness-discrimination": "Workplace Discrimination",
    "illness-selfharm-thought": "Self-Harm Thoughts",
    "illness-selfharm-action": "Acts of Self-Harm",
    "illness-personal-relationships": "Relationship Struggles",
    "illness-humiliation": "Experienced Humiliation",
}

diagnosis_map = {
    "mood_disorder": "Mood Disorder",
    "undiagnosed": "Not Officially Diagnosed",
    "autism/adhd": "Autism / ADHD",
    "ocd": "OCD",
    "anxiety_disorder": "Anxiety Disorder",
    "multiple": "Multiple Disorders",
    "carer": "Caretaker / Family Member",
    "personality_disorder": "Personality Disorder",
    "psychotic_illness": "Psychotic Disorder",
}

event_map = {
    "event-workshops": "Vocational Training Workshops",
    "event-dr-visits": "Psychiatrist Consultations",
    "event-grouptherapy": "Group Therapy Sessions",
    "event-illness-workshops": "Mental Health Awareness Workshops",
    "event-live-QA": "Live Q&A with Mental Health Professionals",
    "event-friendly-gatherings": "Peer Support Social Events",
}

barrier_groups = {
    "Structural & Economic": {
        "color": "#7F1D1D",  # dark burgundy
        "columns": [
            "treatment-abondonment-expenses",
            "treatment-abondonment-resources",
        ],
    },
    "Treatment & Provider-Related": {
        "color": "#1E3A8A",  # deep navy
        "columns": [
            "treatment-abondonment-medication-side-effects",
            "treatment-abondonment-distrust",
            "treatment-abondonment-unprofessional-behavior",
        ],
    },
    "Personal & Social Barriers": {
        "color": "#065F46",  # deep forest green
        "columns": [
            "treatment-abondonment-lackhope",
            "treatment-abondonment-lackknowledge",
            "treatment-abondonment-stigma",
            "treatment-abondonment-support-network",
        ],
    },
}

treatment_label_map = {
    "treatment-abondonment-expenses": "Financial Constraints",
    "treatment-abondonment-medication-side-effects": "Concerns About Medication Side Effects",
    "treatment-abondonment-distrust": "Distrust in Mental Health Providers",
    "treatment-abondonment-lackhope": "Perceived Lack of Treatment Effectiveness",
    "treatment-abondonment-lackknowledge": "Lack of Awareness About Treatment Options",
    "treatment-abondonment-resources": "Limited Access to Mental Health Resources",
    "treatment-abondonment-support-network": "Insufficient Support Network",
    "treatment-abondonment-stigma": "Fear of Stigma or Shame",
    "treatment-abondonment-unprofessional-behavior": "Negative Experiences With Providers",
}


# functions


def set_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 400
    plt.rcParams["font.family"] = "sans-serif"


def load_data(filepath):
    return pd.read_csv(filepath)


def plot_parallel_coordinates(dataframe, diagnosis_name, N):
    """
    N is the number of participants in that diagnostic group
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns

    group_data = dataframe[dataframe["diagnosis"] == diagnosis_name]
    parallel_coordinates(
        group_data[["diagnosis"] + list(numeric_cols)],
        "diagnosis",
        color=["#384860"],
        linewidth=1.5,
        ax=ax,
        alpha=0.9,
    )

    other_data = dataframe[dataframe["diagnosis"] != diagnosis_name]
    parallel_coordinates(
        other_data[["diagnosis"] + list(numeric_cols)],
        "diagnosis",
        color=["#97a6c4"] * len(other_data["diagnosis"].unique()),
        ax=ax,
        alpha=0.2,
    )

    for child in ax.get_children():
        if isinstance(child, plt.Line2D):
            if child.get_xdata()[0] == child.get_xdata()[-1]:
                child.set_color("grey")
                child.set_alpha(0.5)

    ax.set_title(
        f"{diagnosis_map[diagnosis_name]} (N={N})",
        fontsize=10,
        fontweight="bold",
        alpha=0.8,
        pad=10,
    )

    ax.set_xticklabels(
        [illness_impact_map.get(col, col) for col in numeric_cols],
        fontweight="bold",
        fontsize=8,
        alpha=0.8,
    )

    ax.set_yticks(np.arange(20, 101, 20))
    ax.set_yticklabels(np.arange(20, 101, 20), color="grey", fontweight="bold")

    legend_elements = [
        Line2D(
            [0], [0], color="#384860", lw=1, label=f"{diagnosis_map[diagnosis_name]} "
        ),
        Line2D([0], [0], color="#97a6c4", lw=1, label="Other Conditions"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, title_fontsize=10)

    ax.grid(visible=False)
    plt.tight_layout()
    plt.show()


def test_barrier_association(df, barrier1, barrier2):
    """Test if two barriers are significantly associated using Chi-square test"""

    contingency = pd.crosstab(df[barrier1], df[barrier2])
    chi2, p_value, dof, expected = chi2_contingency(contingency)

    n = len(df)
    phi = np.sqrt(chi2 / n)

    return {"chi2": chi2, "p_value": p_value, "phi": phi, "significant": p_value < 0.05}


def plot_participant_diagnoses(stigma, diagnosis_map):
    diagnosis_counts = stigma["diagnosis"].value_counts()
    data = {
        diagnosis_map.get(label, label): count
        for label, count in diagnosis_counts.items()
    }
    n = int(len(stigma))

    custom_colors = [
        "#005f73",
        "#0a9396",
        "#94d2bd",
        "#e9d8a6",
        "#ee9b00",
        "#ca6702",
        "#bb3e03",
        "#ae2012",
        "#9b2226",
    ]

    plt.figure(
        FigureClass=Waffle,
        rows=10,
        values=data,
        title={
            "label": f"Participants Diagnoses (N={n})",
            "loc": "center",
            "fontsize": 10,
            "fontweight": "bold",
        },
        labels=[f"{label} ({count})" for label, count in data.items()],
        legend={"loc": "center left", "bbox_to_anchor": (1.1, 0.5), "fontsize": 8},
        figsize=(10, 6),
        colors=custom_colors[: len(data)],
    )
    plt.tight_layout()
    plt.show()


def plot_impact_heatmap(impact_by_dx, illness_impact_map, diagnosis_map):
    impact_heatmap = impact_by_dx.set_index("diagnosis")
    impact_labels = [illness_impact_map.get(idx, idx) for idx in impact_heatmap.index]
    diagnosis_labels = [diagnosis_map.get(col, col) for col in impact_heatmap.columns]

    fig, ax = plt.subplots(figsize=(20, 10))

    sns.heatmap(
        impact_heatmap.astype(float),
        annot=False,
        cmap="PuBu",
        linewidths=3,
        linecolor="White",
        xticklabels=diagnosis_labels,
        yticklabels=impact_labels,
        cbar=True,
        cbar_kws={"orientation": "horizontal", "pad": 0.08},
        ax=ax,
    )

    plt.xticks(rotation=0, fontweight="bold", fontsize=12)
    plt.yticks(fontweight="bold", fontsize=12)
    plt.ylabel("")
    plt.xlabel("")

    # Set colorbar label if available
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.set_label(
            "(%) of Participants Within Diagnostic Groups", fontweight="bold"
        )

    plt.title("")
    plt.tight_layout()
    plt.show()


def plot_illness_impact(
    illness_impact_percentages, illness_impact_map, total_n: int | None = None
):
    y_labels = illness_impact_percentages["impact of illness"].map(illness_impact_map)
    y_pos = np.arange(len(illness_impact_percentages))

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.hlines(
        y=y_pos,
        xmin=0,
        xmax=illness_impact_percentages["percentage of total participants"],
        color="#3B5998",
        alpha=0.7,
        linewidth=3,
    )
    ax.plot(
        illness_impact_percentages["percentage of total participants"],
        y_pos,
        "o",
        markersize=10,
        color="#3B5998",
    )

    for index in range(len(illness_impact_percentages)):
        x_value = illness_impact_percentages["percentage of total participants"].iloc[
            index
        ]
        y_position = y_pos[index]

        ax.text(
            x_value + 1,
            y_position,
            f"{x_value:.0f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            alpha=0.7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold", fontsize=12)
    ax.set_xlabel(
        "percentage of all participants(%) ",
        fontsize=12,
        fontweight="bold",
        labelpad=15,
    )
    ax.set_ylabel("")
    title = "Reported Effects of Mental Illness on Participants Daily Lives"
    if total_n is not None:
        title += f" (N = {total_n})"
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.grid(visible=False)
    ax.set_xlim(0, 100)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


# --- Additional visualization functions ---


def plot_top_stigma_by_diagnosis(
    stigma_by_diagnosis: pd.Series, diagnosis_map: dict, top_n: int = 5
):
    """Plot top-N diagnostic groups most affected by stigma (average across stigma columns)."""
    series = stigma_by_diagnosis.sort_values(ascending=True)
    top_series = series[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(
        y=range(len(top_series)),
        xmin=0,
        xmax=top_series.values,
        color="#3B5998",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        top_series.values, range(len(top_series)), "o", markersize=8, color="#3B5998"
    )

    for i, v in enumerate(top_series.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontweight="bold")

    ax.set_yticks(range(len(top_series)))
    ax.set_yticklabels(
        [diagnosis_map.get(x, x) for x in top_series.index], fontweight="bold"
    )

    plt.title(
        "Most Affected by Stigma\n\n(Based on discrimination, humiliation, and relationship impacts)",
        fontweight="bold",
        fontsize=12,
        pad=15,
    )
    plt.xlabel("Average Impact Score (%)", fontweight="bold")
    plt.ylabel("")
    sns.despine(top=True, right=True)
    plt.grid(False)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def plot_top_personal_impact_by_diagnosis(
    personal_impact_by_diagnosis: pd.Series, diagnosis_map: dict, top_n: int = 5
):
    """Plot top-N diagnostic groups most affected personally (average across personal impact columns)."""
    series = personal_impact_by_diagnosis.sort_values(ascending=True)
    top_series = series[-top_n:]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(
        y=range(len(top_series)),
        xmin=0,
        xmax=top_series.values,
        color="#3B5998",
        alpha=0.7,
        linewidth=2,
    )
    ax.plot(
        top_series.values, range(len(top_series)), "o", markersize=8, color="#3B5998"
    )

    for i, v in enumerate(top_series.values):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontweight="bold")

    ax.set_yticks(range(len(top_series)))
    ax.set_yticklabels(
        [diagnosis_map.get(x, x) for x in top_series.index], fontweight="bold"
    )

    plt.title(
        "Most Affected Personally\n\n(Based on work performance, self-harm thoughts, and acts of self-harm)",
        fontweight="bold",
        fontsize=12,
        pad=15,
    )
    plt.xlabel("Average Impact Score (%)", fontweight="bold")
    plt.ylabel("")
    sns.despine(top=True, right=True)
    plt.grid(False)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def plot_treatment_barriers(
    treatment_percentages: pd.Series,
    barrier_groups: dict,
    treatment_label_map: dict,
    total_n: int | None = None,
):
    """Plot treatment barriers with grouped annotations and average markers."""
    treatment_percentages = pd.Series(treatment_percentages)
    fig, ax = plt.subplots(figsize=(15, 10))

    current_y = 0
    y_positions = {}
    barrier_colors = {}
    group_averages = {}

    # Build positions, colors, and compute group averages
    for group_name, group_info in barrier_groups.items():
        group_data = treatment_percentages[group_info["columns"]]
        group_avg = group_data.mean()
        group_averages[group_name] = group_avg

        for col in group_data.index:
            y_positions[col] = current_y
            barrier_colors[col] = group_info["color"]
            current_y += 1

        if group_name != list(barrier_groups.keys())[-1]:
            current_y += 1  # spacer between groups

    # Plot each barrier line and point
    for col, value in treatment_percentages.items():
        y = y_positions[col]
        color = barrier_colors[col]
        ax.hlines(y=y, xmin=0, xmax=value, color=color, linewidth=3)
        ax.plot(value, y, "o", color=color, markersize=10)
        ax.text(
            value + 1,
            y,
            f"{value:.0f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            alpha=0.7,
        )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(
        [treatment_label_map[col] for col in y_positions.keys()],
        fontweight="bold",
        fontsize=11,
    )

    # Group labels and average markers
    current_y = 0
    for group_name, group_info in barrier_groups.items():
        group_size = len(group_info["columns"])
        group_middle = current_y + (group_size - 1) / 2

        ax.text(
            -40,
            group_middle,
            group_name,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="#f0f0f0",
                edgecolor="grey",
                boxstyle="round,pad=0.5",
                linewidth=1.5,
                alpha=0.9,
            ),
            rotation=90,
        )

        y_start = current_y - 0.3
        y_end = current_y + group_size - 0.7
        ax.vlines(
            80,
            y_start,
            y_end,
            color=group_info["color"],
            linestyle="--",
            alpha=0.7,
            linewidth=2,
        )
        ax.plot(80, group_middle, marker="D", color=group_info["color"], markersize=5)
        avg = group_averages[group_name]
        ax.text(
            82,
            group_middle,
            f"on average affects {avg:.0f}% of participants",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
            color=group_info["color"],
        )

        current_y += group_size + 1

    ax.set_xlabel(
        "Percentage of Participants (%)", fontsize=12, fontweight="bold", labelpad=15
    )
    title = "Treatment-Seeking Barriers"
    if total_n is not None:
        title += f" (N = {total_n})"
    ax.set_title(title, fontweight="bold", fontsize=14, pad=20)

    ax.grid(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, None)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def plot_event_preferences(
    event_percentages: pd.DataFrame, event_map: dict, total_n: int | None = None
):
    """Plot ranked event preferences as a lollipop chart."""
    y_labels = event_percentages["event"].map(event_map)
    y_pos = np.arange(len(event_percentages))

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.hlines(
        y=y_pos,
        xmin=0,
        xmax=event_percentages["percentage of total participants"],
        color="#3B5998",
        alpha=0.7,
        linewidth=3,
    )
    ax.plot(
        event_percentages["percentage of total participants"],
        y_pos,
        "o",
        markersize=10,
        color="#3B5998",
    )

    for index in range(len(event_percentages)):
        x_value = event_percentages["percentage of total participants"].iloc[index]
        y_position = y_pos[index]
        ax.text(
            x_value + 1,
            y_position,
            f"{x_value:.0f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="black",
            alpha=0.7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold", fontsize=12)
    ax.set_xlabel(
        "Percentage of all participants (%)",
        fontsize=12,
        fontweight="bold",
        labelpad=20,
    )
    ax.set_ylabel("")
    title = "Mental Health Events Participants are More Likely to Attend (if offered free of cost)"
    if total_n is not None:
        title += f" (N = {total_n})"
    ax.set_title(title, fontweight="bold", fontsize=13, pad=20)
    ax.grid(visible=False)
    ax.set_xlim(0, 100)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def plot_barrier_cooccurrence_heatmap(
    co_occurrence: pd.DataFrame, treatment_label_map: dict
):
    """Plot upper-triangular heatmap of barrier co-occurrence (% of participants)."""
    fig, ax = plt.subplots(figsize=(15, 15))
    mask = np.triu(np.ones_like(co_occurrence, dtype=bool), k=1)
    labels = [treatment_label_map[col] for col in co_occurrence.columns]
    sns.heatmap(
        co_occurrence,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="Grays",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.7,
        square=True,
    )
    plt.title(
        "Barrier Co-occurrence (% of Participants)", fontsize=11, fontweight="bold"
    )
    plt.xticks(fontsize=12, rotation=90)
    plt.yticks(fontsize=12)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_hospitalization_sentiment_pie(hospitalized_sentiment_counts: pd.Series):
    """Plot a pie chart of hospitalization sentiment counts."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        hospitalized_sentiment_counts,
        labels=list(hospitalized_sentiment_counts.index.astype(str)),
        autopct="%1.0f%%",
        colors=["#faf3dd", "#c8d5b9", "#8fc0a9"],
        startangle=150,
        textprops={"fontsize": 8, "fontweight": "bold"},
    )
    ax.set_title(
        "Hospitalization Experiences (N={})".format(
            int(hospitalized_sentiment_counts.sum())
        ),
        fontsize=9,
        fontweight="bold",
        pad=10,
    )
    plt.tight_layout()
    plt.show()


# --- Data preparation helpers (for notebook tidying) ---


def compute_impact_by_dx(
    stigma: pd.DataFrame, cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (illness_impact_groupedby_diagnosis, impact_by_dx) in percentage.

    illness_impact_groupedby_diagnosis: mean % for each impact per diagnosis (wide, with 'diagnosis').
    impact_by_dx: transposed layout used by heatmap (impacts as rows, diagnoses as columns).
    """
    if cols is None:
        cols = illness_related_cols
    grouped = stigma.groupby("diagnosis")[cols].mean() * 100
    illness_impact_groupedby_diagnosis = grouped.reset_index()
    impact_by_dx = illness_impact_groupedby_diagnosis.T.reset_index()
    impact_by_dx.columns = impact_by_dx.iloc[0]
    impact_by_dx = impact_by_dx[1:].reset_index(drop=True)
    return illness_impact_groupedby_diagnosis, impact_by_dx


def compute_illness_impact_percentages(
    stigma: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """Return a dataframe of overall illness impact percentages across all participants."""
    if cols is None:
        cols = illness_related_cols
    s = stigma[cols].mean() * 100
    df = pd.DataFrame(
        {
            "impact of illness": s.index,
            "percentage of total participants": s.values,
        }
    ).sort_values(by="percentage of total participants", ascending=False)
    return df


def compute_event_percentages(
    stigma: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """Return event preferences as a dataframe sorted by percentage descending."""
    if cols is None:
        cols = event_related_cols
    s = stigma[cols].mean() * 100
    df = pd.DataFrame(
        {"event": s.index, "percentage of total participants": s.values}
    ).sort_values(by="percentage of total participants", ascending=False)
    return df


def compute_treatment_percentages(
    stigma: pd.DataFrame, cols: list[str] | None = None
) -> pd.Series:
    """Return treatment barrier percentages as a Series (0-100)."""
    if cols is None:
        cols = treatment_related_cols
    return stigma[cols].mean() * 100


def compute_stigma_by_diagnosis(
    stigma: pd.DataFrame, stigma_cols: list[str] | None = None
) -> pd.Series:
    """Average stigma impact across stigma columns per diagnosis (0-100), sorted ascending."""
    if stigma_cols is None:
        stigma_cols = illness_stigma_cols
    series = stigma.groupby("diagnosis")[stigma_cols].mean().mean(axis=1) * 100
    return series.sort_values(ascending=True)


def compute_personal_impact_by_diagnosis(
    stigma: pd.DataFrame, personal_cols: list[str] | None = None
) -> pd.Series:
    """Average personal impact across personal impact columns per diagnosis (0-100), sorted ascending."""
    if personal_cols is None:
        personal_cols = illness_personal_cols
    series = stigma.groupby("diagnosis")[personal_cols].mean().mean(axis=1) * 100
    return series.sort_values(ascending=True)


def compute_co_occurrence(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compute co-occurrence matrix (% of participants) for given boolean/0-1 columns."""
    X = df[cols]
    return (X.T @ X) / len(X) * 100


def get_hospitalized_subset(stigma: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with hospitalization experience and the relevant columns."""
    return stigma[stigma["hospitalization-experience"].notna()][
        ["diagnosis", "hospitalization-experience"]
    ].copy()


def label_hospitalization_sentiment(hospitalized: pd.DataFrame) -> pd.DataFrame:
    """Assign sentiment labels to the provided hospitalized dataframe (dataset-specific indices)."""
    hospitalized = hospitalized.copy()
    hospitalized["sentiment"] = None

    mapping = {
        12: "Mixed",
        101: "Negative",
        113: "Negative",
        114: "Positive",
        128: "Negative",
        133: "Mixed",
        144: "Positive",
        175: "Mixed",
        188: "Mixed",
        224: "Negative",
        229: "Mixed",
        233: "Positive",
        242: "Mixed",
        257: "Mixed",
        263: "Negative",
        271: "Mixed",
        292: "Negative",
        302: "Negative",
        319: "Negative",
    }

    for idx, sentiment in mapping.items():
        if idx in hospitalized.index:
            hospitalized.at[idx, "sentiment"] = sentiment

    return hospitalized


def compute_sentiment_counts(hospitalized: pd.DataFrame) -> pd.Series:
    """Return value counts of the 'sentiment' column."""
    return hospitalized["sentiment"].value_counts()


def print_hospitalized_experiences(hospitalized: pd.DataFrame) -> None:
    """Pretty-print diagnosis and hospitalization experience for each row."""
    for idx, row in hospitalized.iterrows():
        print(f"index:{idx}")
        print(f"Diagnosis: {row['diagnosis']}")
        print(f"Experience: {row['hospitalization-experience']}")
        print("-" * 80)


def compute_significance(
    df: pd.DataFrame,
    columns: list[str],
    label_map: dict[str, str],
    col1_label: str,
    col2_label: str,
) -> pd.DataFrame:
    """Compute chi-square association for all pairs of given columns and return a sorted dataframe."""
    results: list[dict] = []
    for a, b in itertools.combinations(columns, 2):
        stats = test_barrier_association(df, a, b)
        results.append(
            {
                col1_label: label_map.get(a, a),
                col2_label: label_map.get(b, b),
                "Chi2": stats["chi2"],
                "P-value": stats["p_value"],
                "Phi": stats["phi"],
                "Significant": stats["significant"],
            }
        )
    return pd.DataFrame(results).sort_values("P-value")


def print_significant_pairs(
    significance_df: pd.DataFrame,
    col1_label: str,
    col2_label: str,
    *,
    alpha: float = 0.05,
    top_n: int = 10,
) -> None:
    """Print top significant pairs from a significance dataframe."""
    print(
        f"Most Significant {col1_label.split()[0]} {col2_label.split()[0]} Co-occurrences (p < {alpha}):"
    )
    print("=" * 60)
    sig = significance_df[significance_df["Significant"]]
    for _, row in sig.head(top_n).iterrows():
        print(f"{row[col1_label]} ↔ {row[col2_label]}")
        print(
            f"  Chi² = {row['Chi2']:.2f}, p = {row['P-value']:.4f}, φ = {row['Phi']:.3f}"
        )
        print()
