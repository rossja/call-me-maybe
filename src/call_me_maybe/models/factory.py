"""Factory that returns the appropriate ModelBackend for the configured provider."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from call_me_maybe.models.base import ModelBackend

if TYPE_CHECKING:
    from call_me_maybe.config.settings import Settings

logger = logging.getLogger(__name__)


def create_backend(settings: "Settings") -> ModelBackend:
    """
    Instantiate and return the correct :class:`ModelBackend`.

    Parameters
    ----------
    settings:
        Fully resolved :class:`~call_me_maybe.config.settings.Settings` object.

    Returns
    -------
    ModelBackend
        Either a :class:`~call_me_maybe.models.local.LocalMLXBackend`
        (when ``provider == "local"``) or a
        :class:`~call_me_maybe.models.remote.RemoteBackend`
        (when ``provider == "remote"``).

    Raises
    ------
    ValueError
        If ``provider`` is not ``"local"`` or ``"remote"``.
    """
    provider = settings.provider.lower()
    logger.info("Creating model backend: provider=%s", provider)

    if provider == "local":
        from call_me_maybe.models.local import LocalMLXBackend
        return LocalMLXBackend(settings)

    if provider == "remote":
        from call_me_maybe.models.remote import RemoteBackend
        return RemoteBackend(settings)

    raise ValueError(
        f"Unknown provider {provider!r}. Must be 'local' or 'remote'."
    )
