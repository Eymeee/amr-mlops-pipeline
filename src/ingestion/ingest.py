import logging
from pathlib import Path
import kaggle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_ID = "amritpal333/antimicrobial-resistance-data"
RAW_DIR    = Path("data/raw")


def download_dataset(dataset_id: str = DATASET_ID, output_dir: Path = RAW_DIR) -> Path:
    """Télécharge le dataset depuis Kaggle et le sauvegarde dans data/raw."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Téléchargement du dataset : {dataset_id}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset=dataset_id,
        path=output_dir,
        unzip=True,
        quiet=False,
    )
    logger.info(f"Dataset téléchargé dans : {output_dir}")
    return output_dir


def inspect_dataset(raw_dir: Path = RAW_DIR) -> dict:
    """Inspecte la structure du dataset et retourne un résumé."""
    xlsx_files = list(raw_dir.rglob("*.xlsx"))
    png_files  = list(raw_dir.rglob("*.png"))

    if not xlsx_files:
        raise FileNotFoundError("Aucun fichier .xlsx trouvé dans data/raw/")

    # Extraire les années et microbes disponibles
    years    = sorted(set(
        f.parts[f.parts.index("Downloadables") - 1]
        for f in xlsx_files
        if "Downloadables" in f.parts
    ))
    microbes = sorted(set(f.parent.name for f in xlsx_files))

    summary = {
        "total_xlsx" : len(xlsx_files),
        "total_png"  : len(png_files),
        "years"      : years,
        "microbes"   : microbes,
    }

    logger.info(f"Total fichiers .xlsx : {summary['total_xlsx']}")
    logger.info(f"Années disponibles   : {summary['years']}")
    logger.info(f"Microbes disponibles : {summary['microbes']}")

    return summary


if __name__ == "__main__":
    # Si déjà téléchargé, on skip le download
    if not list(RAW_DIR.rglob("*.xlsx")):
        download_dataset()
    else:
        logger.info("Dataset déjà présent, skip du téléchargement.")

    inspect_dataset()