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

    Supports both single-provider (all STT/LLM/TTS on same backend) and
    mixed-provider (each component routed independently) configurations.

    Parameters
    ----------
    settings:
        Fully resolved :class:`~call_me_maybe.config.settings.Settings` object.

    Returns
    -------
    ModelBackend
        Either a single :class:`~call_me_maybe.models.local.LocalMLXBackend`,
        :class:`~call_me_maybe.models.remote.RemoteBackend`, or a
        :class:`~call_me_maybe.models.composite.CompositeBackend` if components
        use different providers.

    Raises
    ------
    ValueError
        If any component provider is not ``"local"`` or ``"remote"``.
    """
    from call_me_maybe.models.local import LocalMLXBackend
    from call_me_maybe.models.remote import RemoteBackend

    stt_p = settings.component_provider("stt")
    llm_p = settings.component_provider("llm")
    tts_p = settings.component_provider("tts")

    logger.info(
        "Creating model backend: stt=%s llm=%s tts=%s",
        stt_p,
        llm_p,
        tts_p,
    )

    # Validate all providers are valid
    for p in (stt_p, llm_p, tts_p):
        if p not in ("local", "remote"):
            raise ValueError(f"Unknown provider {p!r}. Must be 'local' or 'remote'.")

    # All components use same provider: return single backend for efficiency
    if stt_p == llm_p == tts_p == "local":
        return LocalMLXBackend(settings)

    if stt_p == llm_p == tts_p == "remote":
        return RemoteBackend(settings)

    # Mixed providers: build composite backend
    from call_me_maybe.models.composite import CompositeBackend

    local = LocalMLXBackend(settings) if "local" in (stt_p, llm_p, tts_p) else None

    def _backend(component: str, provider: str) -> ModelBackend:
        if provider == "local":
            assert local is not None
            return local
        return RemoteBackend(
            settings,
            base_url=settings.component_base_url(component),
            api_key=settings.component_api_key(component),
        )

    return CompositeBackend(
        stt=_backend("stt", stt_p),
        llm=_backend("llm", llm_p),
        tts=_backend("tts", tts_p),
    )
