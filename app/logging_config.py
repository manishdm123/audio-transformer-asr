from __future__ import annotations

import logging


_HANDLER_MARKER = "audio2text_timestamped"


def get_logger(component: str) -> logging.Logger:
    base_logger = logging.getLogger("audio2text")
    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in base_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        setattr(handler, _HANDLER_MARKER, True)
        base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)
        base_logger.propagate = False
    return logging.getLogger(f"audio2text.{component}")
