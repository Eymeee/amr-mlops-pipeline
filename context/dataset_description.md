# Dataset Analysis — Antibiotic Resistance Tracking Dataset

## Dataset Structure

- **Format**: CSV
- **Rows**: 2,200 (patient records)
- **Columns**: 12
- **Missing values**: none

| Column | Type | Possible Values |
|---|---|---|
| `Patient_ID` | ID | P0001 → P2200 |
| `Age` | Numeric | 1 → 90 |
| `Gender` | Categorical | Male / Female |
| `Specimen_Type` | Categorical | Blood, Urine, Sputum, Wound swab, Stool |
| `Amoxicillin` | Categorical | Sensitive / Intermediate / Resistant |
| `Ciprofloxacin` | Categorical | Sensitive / Intermediate / Resistant |
| `Meropenem` | Categorical | Sensitive / Intermediate / Resistant |
| `Vancomycin` | Categorical | Sensitive / Intermediate / Resistant |
| `Colistin` | Categorical | Sensitive / Intermediate / Resistant |
| `Test_Method` | Categorical | Automated System / MIC / Disc Diffusion |
| `Resistance_Genes` | Categorical | KPC, OXA-48, VIM, NDM-1, None |
| `Outcome` | **Target** | **Recovered / ICU / Deceased** |

---

## Prediction Target

### Chosen target: `Outcome`

We predict the **clinical outcome of the patient** based on their antibiotic resistance profile.
This is a **multi-class classification problem with 3 classes**: `Recovered`, `ICU`, `Deceased`.

### Features (X)

```
- Age               → numeric
- Gender            → categorical (Male / Female)
- Specimen_Type     → categorical (Blood, Urine, Sputum, Wound swab, Stool)
- Amoxicillin       → categorical (Sensitive / Intermediate / Resistant)
- Ciprofloxacin     → categorical (Sensitive / Intermediate / Resistant)
- Meropenem         → categorical (Sensitive / Intermediate / Resistant)
- Vancomycin        → categorical (Sensitive / Intermediate / Resistant)
- Colistin          → categorical (Sensitive / Intermediate / Resistant)
- Test_Method       → categorical (Automated System / MIC / Disc Diffusion)
- Resistance_Genes  → categorical (KPC, OXA-48, VIM, NDM-1, None)
```

### Target (y)

```
Outcome → Recovered / ICU / Deceased
```

### Recommended Model

**LightGBM** — multi-class classification, ideal for:
- Mixed features (numeric + categorical)
- Small dataset size (2,200 rows)
- No GPU required (100% CPU-compatible)

### Evaluation Metrics

| Metric | Target | Justification |
|---|---|---|
| **F1-Score Macro** | > 0.85 | Balance across all 3 classes |
| **AUC-ROC OvR** | > 0.90 | Multi-class discriminability |
| **Accuracy** | > 80% | Overall performance measure |
| **Confusion Matrix** | — | Detailed analysis of ICU vs Deceased errors |

### Clinical Value

Predicting the `Outcome` of a patient from their antibiotic resistance profile enables:
- Anticipating critical cases requiring **ICU** admission
- Identifying patients at risk of **death**
- Guiding therapeutic decisions based on the genetic resistance profile
