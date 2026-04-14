import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging() -> logging.Logger:
    """
    Configure le système de logs pour toute l'application.
    Les logs sont écrits dans un fichier app/logs/app.log
    avec rotation pour éviter qu'il grossisse trop.
    """
    log_dir = Path("app/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger