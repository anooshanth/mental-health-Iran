import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import parallel_coordinates
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency, fisher_exact
import itertools


illness_related_cols = [
    'illness-performance', 'illness-discrimination',
    'illness-selfharm-thought', 'illness-selfharm-action',
    'illness-personal-relationships', 'illness-humiliation'
]

treatment_related_cols = [
    'treatment-abondonment-expenses',
    'treatment-abondonment-medication-side-effects',
    'treatment-abondonment-distrust', 'treatment-abondonment-lackhope',
    'treatment-abondonment-lackknowledge',
    'treatment-abondonment-resources',
    'treatment-abondonment-support-network', 'treatment-abondonment-stigma',
    'treatment-abondonment-unprofessional-behavior'
]

event_related_cols = [
    'event-workshops', 'event-dr-visits',
    'event-grouptherapy', 'event-illness-workshops', 'event-live-QA',
    'event-friendly-gatherings'
]

illness_stigma_cols = [
    'illness-humiliation', 'illness-discrimination', 'illness-personal-relationships'
]

illness_personal_cols = [
    'illness-performance', 'illness-selfharm-thought', 'illness-selfharm-action'
]

illness_impact_map = {
    'illness-performance': 'Work Performance Issues',
    'illness-discrimination': 'Workplace Discrimination',
    'illness-selfharm-thought': 'Self-Harm Thoughts',
    'illness-selfharm-action': 'Acts of Self-Harm',
    'illness-personal-relationships': 'Relationship Struggles',
    'illness-humiliation': 'Experienced Humiliation'
}

diagnosis_map = {
    'mood_disorder': 'Mood Disorder',
    'undiagnosed': 'Not Officially Diagnosed',
    'autism/adhd': 'Autism / ADHD',
    'ocd': 'OCD',
    'anxiety_disorder': 'Anxiety Disorder',
    'multiple': 'Multiple Disorders',
    'carer': 'Caretaker / Family Member',
    'personality_disorder': 'Personality Disorder',
    'psychotic_illness': 'Psychotic Disorder'
}

event_map = {
    'event-workshops': 'Vocational Training Workshops',
    'event-dr-visits': 'Psychiatrist Consultations',
    'event-grouptherapy': 'Group Therapy Sessions',
    'event-illness-workshops': 'Mental Health Awareness Workshops',
    'event-live-QA': 'Live Q&A with Mental Health Professionals',
    'event-friendly-gatherings': 'Peer Support Social Events'
}

barrier_groups = {
    'Structural & Economic': {
        'color': '#E63946',  # red
        'columns': ['treatment-abondonment-expenses', 'treatment-abondonment-resources']
    },
    'Treatment & Provider-Related': {
        'color': '#457B9D',
        'columns': ['treatment-abondonment-medication-side-effects',
                   'treatment-abondonment-distrust',
                   'treatment-abondonment-unprofessional-behavior']
    },
    'Personal & Social Barriers': {
        'color': '#2A9D8F',
        'columns': ['treatment-abondonment-lackhope',
                   'treatment-abondonment-lackknowledge',
                   'treatment-abondonment-stigma',
                   'treatment-abondonment-support-network']
    }
}

treatment_label_map = {
    'treatment-abondonment-expenses': 'Financial Constraints',
    'treatment-abondonment-medication-side-effects': 'Concerns About Medication Side Effects',
    'treatment-abondonment-distrust': 'Distrust in Mental Health Providers',
    'treatment-abondonment-lackhope': 'Perceived Lack of Treatment Effectiveness',
    'treatment-abondonment-lackknowledge': 'Lack of Awareness About Treatment Options',
    'treatment-abondonment-resources': 'Limited Access to Mental Health Resources',
    'treatment-abondonment-support-network': 'Insufficient Support Network',
    'treatment-abondonment-stigma': 'Fear of Stigma or Shame',
    'treatment-abondonment-unprofessional-behavior': 'Negative Experiences With Providers'
}



# functions

def set_plot_style():
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['font.family'] = 'sans-serif'

def load_data(filepath):
    return pd.read_csv(filepath)

def plot_parallel_coordinates(dataframe, diagnosis_name, N):
    """
    N is the number of participants in that diagnostic group
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns

    group_data = dataframe[dataframe['diagnosis'] == diagnosis_name]
    parallel_coordinates(
        group_data[['diagnosis'] + list(numeric_cols)],
        'diagnosis',
        color=['#384860'],
        linewidth=1.5,
        ax=ax,
        alpha=0.9
    )

    other_data = dataframe[dataframe['diagnosis'] != diagnosis_name]
    parallel_coordinates(
        other_data[['diagnosis'] + list(numeric_cols)],
        'diagnosis',
        color=['#97a6c4'] * len(other_data['diagnosis'].unique()),
        ax=ax,
        alpha=0.2
    )


    for child in ax.get_children():
        if isinstance(child, plt.Line2D):
            if child.get_xdata()[0] == child.get_xdata()[-1]: 
                child.set_color('grey')
                child.set_alpha(0.5)

    ax.set_title(f"{diagnosis_map[diagnosis_name]} (N={N})", fontsize=10, fontweight='bold', alpha=0.8, pad=10)
    
    ax.set_xticklabels(
        [illness_impact_map.get(col, col) for col in numeric_cols],
        fontweight='bold',
        fontsize=8,
        alpha=0.8
    )

    ax.set_yticks(np.arange(20, 101, 20))
    ax.set_yticklabels(np.arange(20, 101, 20), color='grey', fontweight='bold')

    legend_elements = [
        Line2D([0], [0], color='#384860', lw=1, label=f'{diagnosis_map[diagnosis_name]} '),
        Line2D([0], [0], color='#97a6c4', lw=1, label='Other Conditions')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, title_fontsize=10)

    ax.grid(visible=False)
    plt.tight_layout()
    plt.show()

def test_barrier_association(df, barrier1, barrier2):
    """Test if two barriers are significantly associated using Chi-square test"""

    contingency = pd.crosstab(df[barrier1], df[barrier2])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    

    n = len(df)
    phi = np.sqrt(chi2 / n)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'phi': phi,
        'significant': p_value < 0.05
    }
