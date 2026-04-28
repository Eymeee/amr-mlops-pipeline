# Dataset Analysis — Antibiotic Resistance Tracking Dataset

## Structure du Dataset

- **Format** : CSV
- **Lignes** : 2 200 (enregistrements patients)
- **Colonnes** : 12
- **Valeurs manquantes** : aucune

| Colonne | Type | Valeurs possibles |
|---|---|---|
| `Patient_ID` | ID | P0001 → P1000 |
| `Age` | Numérique | 1 → 90 |
| `Gender` | Catégoriel | Male / Female |
| `Specimen_Type` | Catégoriel | Blood, Urine, Sputum, Wound swab, Stool |
| `Amoxicillin` | Catégoriel | Sensitive / Intermediate / Resistant |
| `Ciprofloxacin` | Catégoriel | Sensitive / Intermediate / Resistant |
| `Meropenem` | Catégoriel | Sensitive / Intermediate / Resistant |
| `Vancomycin` | Catégoriel | Sensitive / Intermediate / Resistant |
| `Colistin` | Catégoriel | Sensitive / Intermediate / Resistant |
| `Test_Method` | Catégoriel | Automated System / MIC / Disc Diffusion |
| `Resistance_Genes` | Catégoriel | KPC, OXA-48, VIM, NDM-1, None |
| `Outcome` | **Cible** | **Recovered / ICU / Deceased** |

---

## Cible de Prédiction

### Cible choisie : `Outcome`

On prédit le **devenir clinique du patient** à partir de son profil de résistance aux antibiotiques.
C'est une **classification multi-classe à 3 classes** : `Recovered`, `ICU`, `Deceased`.

### Features (X)

```
- Age                  → numérique
- Gender               → catégoriel (Male / Female)
- Specimen_Type        → catégoriel (Blood, Urine, Sputum, Wound swab, Stool)
- Amoxicillin          → catégoriel (Sensitive / Intermediate / Resistant)
- Ciprofloxacin        → catégoriel (Sensitive / Intermediate / Resistant)
- Meropenem            → catégoriel (Sensitive / Intermediate / Resistant)
- Vancomycin           → catégoriel (Sensitive / Intermediate / Resistant)
- Colistin             → catégoriel (Sensitive / Intermediate / Resistant)
- Test_Method          → catégoriel (Automated System / MIC / Disc Diffusion)
- Resistance_Genes     → catégoriel (KPC, OXA-48, VIM, NDM-1, None)
```

### Cible (y)

```
Outcome → Recovered / ICU / Deceased
```

### Modèle recommandé

**LightGBM** — classification multi-classe, idéal pour :
- Features mixtes (numériques + catégorielles)
- Faible volumétrie (2 200 lignes)
- Pas de GPU nécessaire (100% CPU)

### Métriques d'évaluation

| Métrique | Objectif | Justification |
|---|---|---|
| **F1-Score Macro** | > 0.85 | Équilibre entre les 3 classes |
| **AUC-ROC OvR** | > 0.90 | Discriminabilité multi-classe |
| **Accuracy** | > 80% | Mesure globale |
| **Matrice de confusion** | — | Visualiser Recovered vs Deceased notamment |

### Valeur clinique

Prédire l'`Outcome` d'un patient à partir de son profil de résistance aux antibiotiques permet :
- D'anticiper les cas critiques nécessitant une admission en **ICU**
- D'identifier les patients à risque de **décès**
- D'orienter les décisions thérapeutiques en fonction du profil de résistance génétique
