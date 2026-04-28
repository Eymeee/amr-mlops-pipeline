# Référence — Concepts du Dataset Antibiotic Resistance Tracking

## 🦠 La résistance aux antibiotiques

Un antibiotique est un médicament conçu pour **tuer ou bloquer la prolifération des bactéries**.
La résistance apparaît quand une bactérie **mute ou acquiert des gènes** qui lui permettent de survivre malgré l'antibiotique — elle le neutralise, le pompe hors de ses cellules, ou modifie sa cible.

**D'où vient-elle ?**
- **Utilisation excessive** des antibiotiques (automédication, élevage intensif)
- **Pression sélective** : les bactéries sensibles meurent, les résistantes survivent et se multiplient
- **Transfert horizontal de gènes** : une bactérie résistante peut "donner" ses gènes de résistance à d'autres bactéries

En 2019, l'AMR (Antimicrobial Resistance) a causé ~5 millions de décès indirectement — c'est une des plus grandes crises sanitaires mondiales.

---

## 💊 Les 5 antibiotiques du dataset

Ils représentent des **lignes de traitement successives** face aux infections graves :

| Antibiotique | Famille | Rôle clinique |
|---|---|---|
| **Amoxicillin** | Pénicilline | 1ère ligne — antibiotique de base, le plus prescrit au monde |
| **Ciprofloxacin** | Fluoroquinolone | 2ème ligne — infections urinaires, respiratoires |
| **Meropenem** | Carbapénème | 3ème ligne — infections graves, résistantes aux autres |
| **Vancomycin** | Glycopeptide | Dernier recours — infections à bactéries Gram+ (MRSA) |
| **Colistin** | Polymyxine | Ultime recours — quand tout le reste a échoué |

> L'ordre est important : si le patient est résistant à **Amoxicillin**, on passe à Ciprofloxacin, puis Meropenem, etc. Un patient résistant à **Colistin** n'a pratiquement plus d'options thérapeutiques.

---

## 🏥 La colonne `Outcome`

C'est le **résultat clinique du patient** — pas le résultat du test, mais ce qui lui est arrivé médicalement après le diagnostic et la prise en charge.

Chaque ligne = un patient hospitalisé avec une infection bactérienne identifiée. On teste sa bactérie contre les 5 antibiotiques, et `Outcome` dit **comment s'est terminée son hospitalisation**.

| Valeur | Signification |
|---|---|
| **Recovered** | Patient guéri et sorti de l'hôpital |
| **ICU** | Patient transféré en réanimation — état grave, pronostic incertain |
| **Deceased** | Patient décédé des suites de l'infection |

---

## 📋 Explication des colonnes

### `Specimen_Type` — Type de prélèvement biologique
D'où vient l'échantillon utilisé pour identifier la bactérie :

| Valeur | Description |
|---|---|
| **Blood** | Hémoculture — bactérie dans le sang → septicémie, très grave |
| **Urine** | ECBU — infection urinaire (cystite, pyélonéphrite) |
| **Sputum** | Crachat — infection respiratoire (pneumonie, bronchite) |
| **Wound swab** | Écouvillon de plaie — infection de blessure, post-opératoire |
| **Stool** | Selles — infection digestive (gastro-entérite bactérienne) |

---

### `Amoxicillin / Ciprofloxacin / Meropenem / Vancomycin / Colistin`
Résultat du test de sensibilité de la bactérie du patient à chaque antibiotique :

| Valeur | Signification clinique |
|---|---|
| **Sensitive** | L'antibiotique **tue** la bactérie → traitement efficace |
| **Intermediate** | Efficacité **partielle ou douteuse** → à éviter si possible |
| **Resistant** | L'antibiotique **ne fait rien** → traitement inutile, dangereux à prescrire |

---

### `Test_Method` — Méthode de test utilisée en laboratoire

| Valeur | Description |
|---|---|
| **MIC** | Minimum Inhibitory Concentration — mesure la dose minimale d'antibiotique nécessaire pour bloquer la bactérie. Très précis, résultat quantitatif |
| **Disc Diffusion** | Disques imbibés d'antibiotiques posés sur une culture bactérienne — on mesure la zone d'inhibition. Méthode visuelle classique |
| **Automated System** | Machine automatisée (ex: VITEK) — rapide, standard dans les hôpitaux modernes |

---

### `Resistance_Genes` — Gènes de résistance identifiés
Gènes spécifiques que possède la bactérie pour résister — expliquent *pourquoi* elle est résistante :

| Valeur | Nom complet | Ce qu'il fait |
|---|---|---|
| **KPC** | Klebsiella pneumoniae Carbapenemase | Détruit les carbapénèmes (dont Meropenem) — très répandu |
| **NDM-1** | New Delhi Metallo-β-lactamase | Détruit presque tous les antibiotiques β-lactamines — très dangereux |
| **OXA-48** | Oxacillinase-48 | Résistance aux carbapénèmes, difficile à détecter |
| **VIM** | Verona Integron-encoded Metallo-β-lactamase | Résistance large spectre, fréquent en Europe |
| **None** | Aucun gène identifié | Résistance possible par d'autres mécanismes non génétiques |

---

## 🔁 Résumé global

**Chaque ligne du dataset = un patient hospitalisé** avec une infection bactérienne, dont on a :
1. Identifié la source de l'infection (`Specimen_Type`)
2. Testé la résistance de sa bactérie aux 5 antibiotiques
3. Utilisé une méthode de test standardisée (`Test_Method`)
4. Séquencé les gènes de résistance (`Resistance_Genes`)
5. Suivi l'évolution jusqu'à la sortie ou le décès (`Outcome`)

**Objectif ML :** prédire l'`Outcome` (Recovered / ICU / Deceased) à partir de tous ces paramètres.
