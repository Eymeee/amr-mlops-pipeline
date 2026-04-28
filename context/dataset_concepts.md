# Reference — Antibiotic Resistance Tracking Dataset Concepts

## 🦠 What is Antibiotic Resistance?

An antibiotic is a drug designed to **kill or stop the growth of bacteria**.
Resistance occurs when a bacterium **mutates or acquires genes** that allow it to survive despite the antibiotic — it can neutralize it, pump it out of its cells, or modify its target.

**Where does it come from?**
- **Overuse of antibiotics** (self-medication, intensive livestock farming)
- **Selective pressure**: sensitive bacteria die, resistant ones survive and multiply
- **Horizontal gene transfer**: a resistant bacterium can "share" its resistance genes with other bacteria

In 2019, AMR (Antimicrobial Resistance) was indirectly responsible for ~5 million deaths worldwide — making it one of the greatest public health crises of our time.

---

## 💊 The 5 Antibiotics in the Dataset

They represent **successive lines of treatment** for serious bacterial infections:

| Antibiotic | Family | Clinical Role |
|---|---|---|
| **Amoxicillin** | Penicillin | 1st line — the most commonly prescribed antibiotic worldwide |
| **Ciprofloxacin** | Fluoroquinolone | 2nd line — urinary and respiratory tract infections |
| **Meropenem** | Carbapenem | 3rd line — severe infections resistant to previous antibiotics |
| **Vancomycin** | Glycopeptide | Last resort — Gram-positive bacterial infections (MRSA) |
| **Colistin** | Polymyxin | Ultimate last resort — when everything else has failed |

> Order matters: if the patient is resistant to **Amoxicillin**, we move to Ciprofloxacin, then Meropenem, etc. A patient resistant to **Colistin** has virtually no remaining treatment options.

---

## 🏥 The `Outcome` Column

This represents the **clinical outcome of the patient** — not the result of the lab test, but what **actually happened medically** after diagnosis and treatment.

Each row = a hospitalized patient with an identified bacterial infection. The bacteria is tested against the 5 antibiotics, and `Outcome` indicates **how the hospitalization ended**.

| Value | Meaning |
|---|---|
| **Recovered** | Patient healed and discharged from hospital |
| **ICU** | Patient transferred to intensive care unit — critical condition, uncertain prognosis |
| **Deceased** | Patient died as a result of the infection |

---

## 📋 Column Descriptions

### `Specimen_Type` — Biological Sample Type
Indicates where the sample used to identify the bacteria was taken from:

| Value | Description |
|---|---|
| **Blood** | Blood culture — bacteria in the bloodstream → septicemia, very serious |
| **Urine** | Urine culture — urinary tract infection (cystitis, pyelonephritis) |
| **Sputum** | Sputum sample — respiratory infection (pneumonia, bronchitis) |
| **Wound swab** | Wound swab — wound or post-surgical infection |
| **Stool** | Stool sample — digestive infection (bacterial gastroenteritis) |

---

### `Amoxicillin / Ciprofloxacin / Meropenem / Vancomycin / Colistin`
Result of the susceptibility test of the patient's bacteria against each antibiotic:

| Value | Clinical Meaning |
|---|---|
| **Sensitive** | The antibiotic **kills** the bacteria → effective treatment |
| **Intermediate** | **Partial or uncertain** effectiveness → avoid if possible |
| **Resistant** | The antibiotic **has no effect** → treatment is useless and potentially dangerous to prescribe |

---

### `Test_Method` — Laboratory Testing Method

| Value | Description |
|---|---|
| **MIC** | Minimum Inhibitory Concentration — measures the minimum dose of antibiotic needed to inhibit bacterial growth. Very precise, quantitative result |
| **Disc Diffusion** | Antibiotic-soaked discs placed on a bacterial culture — the inhibition zone around each disc is measured. Classic visual method |
| **Automated System** | Automated machine (e.g., VITEK) — fast, standard in modern hospitals |

---

### `Resistance_Genes` — Identified Resistance Genes
Specific genes the bacteria carries to resist antibiotics — they explain *why* it is resistant:

| Value | Full Name | What It Does |
|---|---|---|
| **KPC** | Klebsiella pneumoniae Carbapenemase | Destroys carbapenems (including Meropenem) — very widespread |
| **NDM-1** | New Delhi Metallo-β-lactamase | Destroys almost all β-lactam antibiotics — extremely dangerous |
| **OXA-48** | Oxacillinase-48 | Carbapenem resistance, difficult to detect |
| **VIM** | Verona Integron-encoded Metallo-β-lactamase | Broad-spectrum resistance, common in Europe |
| **None** | No gene identified | Resistance may be due to other non-genetic mechanisms |

---

## 🔁 Summary

**Each row in the dataset = a hospitalized patient** with a bacterial infection, for whom clinicians have:
1. Identified the source of the infection (`Specimen_Type`)
2. Tested the bacteria's resistance to the 5 antibiotics
3. Used a standardized testing method (`Test_Method`)
4. Sequenced the resistance genes (`Resistance_Genes`)
5. Tracked the evolution until discharge or death (`Outcome`)

**ML Goal:** predict the `Outcome` (Recovered / ICU / Deceased) from all these parameters.
