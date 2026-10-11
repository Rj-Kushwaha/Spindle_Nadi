"""Bilingual status announcement text shared by the UI and tests."""

_SPOKEN_MESSAGES = {
    "NORMAL": (
        {"text": "All is well. The machine is operating normally.", "lang": "en-IN"},
        {"text": "सब ठीक है। मशीन सामान्य रूप से चल रही है।", "lang": "hi-IN"},
    ),
    "WARNING": (
        {"text": "Warning. Please inspect the machine soon.", "lang": "en-IN"},
        {"text": "चेतावनी। कृपया मशीन की जल्द जाँच करें।", "lang": "hi-IN"},
    ),
    "CRITICAL": (
        {"text": "Critical alert. Urgent maintenance attention is needed.", "lang": "en-IN"},
        {"text": "गंभीर चेतावनी। मशीन की तुरंत जाँच आवश्यक है।", "lang": "hi-IN"},
    ),
}


def spoken_status_message(status: str, machine_id: str, risk: float) -> str:
    """Create the visible, bilingual announcement for a known status."""
    if status not in _SPOKEN_MESSAGES:
        raise ValueError(f"Unsupported machine status: {status}")
    english, hindi = _SPOKEN_MESSAGES[status]
    return (
        f"{machine_id}: {status} at {risk:.0%} estimated risk. "
        f"{english['text']} {hindi['text']}"
    )
