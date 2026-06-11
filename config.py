"""
config.py
---------
Semua konstanta, mapping label, dan warna kelas.
Tidak ada logika — hanya data.
"""

from pathlib import Path

# ── Path ───────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODEL_PATH     = BASE_DIR / "best_model_D.pt"
LABEL_MAP_PATH = BASE_DIR / "label_map.json"

# ── Hyperparameter model ───────────────────────────────────────────────────────
ROBERTA_NAME = "roberta-base"
MAX_LENGTH   = 56
NUM_CLASSES  = 10
HIDDEN_DIM   = 256
DROPOUT      = 0.3

# ── Label kelas ────────────────────────────────────────────────────────────────
CLASS_LABELS: list[str] = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "requests_or_urgent_needs",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
]

CLASS_LABELS_DISPLAY: dict[str, str] = {
    "caution_and_advice":                    "Caution & Advice",
    "displaced_people_and_evacuations":      "Displaced / Evacuations",
    "infrastructure_and_utility_damage":     "Infrastructure Damage",
    "injured_or_dead_people":               "Injured / Dead People",
    "missing_or_found_people":              "Missing / Found People",
    "not_humanitarian":                      "Not Humanitarian",
    "other_relevant_information":            "Other Relevant Info",
    "requests_or_urgent_needs":             "Requests / Urgent Needs",
    "rescue_volunteering_or_donation_effort":"Rescue / Donation",
    "sympathy_and_support":                 "Sympathy & Support",
}

CLASS_COLORS: dict[str, str] = {
    "caution_and_advice":                    "#F39C12",
    "displaced_people_and_evacuations":      "#8E44AD",
    "infrastructure_and_utility_damage":     "#2980B9",
    "injured_or_dead_people":               "#E74C3C",
    "missing_or_found_people":              "#C0392B",
    "not_humanitarian":                      "#7F8C8D",
    "other_relevant_information":            "#27AE60",
    "requests_or_urgent_needs":             "#D35400",
    "rescue_volunteering_or_donation_effort":"#16A085",
    "sympathy_and_support":                 "#2C3E50",
}

# ── Contoh tweet untuk demo ────────────────────────────────────────────────────
EXAMPLE_TWEETS: list[str] = [
    "People are reported missing after the devastating flood hit the eastern district",
    "Please donate food and clothes to the victims of the earthquake in the shelter",
    "Road to downtown completely destroyed, bridges collapsed after the storm",
    "Our thoughts and prayers go out to everyone affected by this tragedy",
    "Authorities warn residents to stay indoors as the storm approaches the coast",
    "Has anyone seen my family near the flood area, we lost contact since this morning please help",
    "Survivors are being moved to higher ground but many are still unaccounted for",
]
