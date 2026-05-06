"""Logger setup utilities."""
import logging
import os

logging.getLogger("qwen_vl_utils").setLevel(logging.ERROR)

_LOGGER_NAME = "video_icl"
_LOG_PATH_ENV = "VIDEO_ICL_LOG_PATH"


def set_logger(log_path: str | None = None):
    """Create and configure a logger."""
    resolved_log_path = log_path or os.environ.get(_LOG_PATH_ENV, "logs/inference.log")
    log_dir = os.path.dirname(resolved_log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    if getattr(logger, "_configured_log_path", None) == resolved_log_path and logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = logging.FileHandler(resolved_log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger._configured_log_path = resolved_log_path
    return logger
