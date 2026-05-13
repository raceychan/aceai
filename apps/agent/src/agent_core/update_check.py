"""AceAI package update checks."""

import asyncio
import json
from urllib.error import URLError
from urllib.request import urlopen

import aceai
from msgspec import Struct

PYPI_PROJECT_JSON_URL = "https://pypi.org/pypi/aceai/json"


class UpdateCheckResult(Struct, frozen=True, kw_only=True):
    current_version: str
    latest_version: str

    @property
    def has_update(self) -> bool:
        return _version_parts(self.latest_version) > _version_parts(
            self.current_version
        )


async def check_for_updates() -> UpdateCheckResult | None:
    current_version = aceai.__version__
    latest_version = await _fetch_latest_package_version()
    if latest_version is None:
        return None
    return UpdateCheckResult(
        current_version=current_version,
        latest_version=latest_version,
    )


async def _fetch_latest_package_version() -> str | None:
    return await asyncio.to_thread(_fetch_latest_package_version_sync)


def _fetch_latest_package_version_sync() -> str | None:
    try:
        with urlopen(PYPI_PROJECT_JSON_URL, timeout=2.0) as response:
            payload = json.loads(response.read().decode())
    except (URLError, TimeoutError):
        return None
    if type(payload) is not dict:
        raise TypeError("PyPI project payload must be a mapping")
    info = payload["info"]
    if type(info) is not dict:
        raise TypeError("PyPI project info must be a mapping")
    version = info["version"]
    if type(version) is not str:
        raise TypeError("PyPI project version must be str")
    return version


def _version_parts(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError("AceAI version must use x.y.z format")
    return (int(parts[0]), int(parts[1]), int(parts[2]))
