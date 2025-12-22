import logging
from typing import Dict, Any


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Создаёт и настраивает корневой логгер проекта ``kmeans``.

    :param level: минимальный уровень логирования
    :return: настроенный экземпляр :class:`logging.Logger`
    """
    logger = logging.getLogger("kmeans")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Чтобы сообщения не дублировались через root-логгер
    logger.propagate = False

    return logger


def format_dataset_prefix(meta: Dict[str, Any]) -> str:
    """
    Формирует текстовый префикс для логов по информации о датасете.

    Ожидается словарь из summary с ключами ``N``, ``D``, ``K`` и опциональным
    ``purpose``.
    """
    return (
        f"[N={meta['N']} D={meta['D']} K={meta['K']} "
        f"purpose={meta.get('purpose', 'base')}]"
    )
