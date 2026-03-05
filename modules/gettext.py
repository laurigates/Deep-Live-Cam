import json
import re
from pathlib import Path

# Allowlist: 1–10 chars, letters/digits/hyphens only (covers BCP-47 tags like pt-br)
_LANGUAGE_CODE_RE = re.compile(r"^[a-zA-Z0-9-]{1,10}$")


class LanguageManager:
    def __init__(self, default_language="en"):
        self.current_language = default_language
        self.translations = {}
        self.load_language(default_language)

    def load_language(self, language_code) -> bool:
        """load language file"""
        if language_code == "en":
            return True
        if not isinstance(language_code, str) or not _LANGUAGE_CODE_RE.match(language_code):
            print(f"Invalid language code: {language_code!r}")
            return False
        locales_dir = (Path(__file__).parent.parent / "locales").resolve()
        file_path = (locales_dir / f"{language_code}.json").resolve()
        if not str(file_path).startswith(str(locales_dir)):
            print(f"Invalid language code: {language_code!r}")
            return False
        try:
            with open(file_path, encoding="utf-8") as file:
                self.translations = json.load(file)
            self.current_language = language_code
            return True
        except FileNotFoundError:
            print(f"Language file not found: {language_code}")
            return False

    def _(self, key, default=None) -> str:
        """get translate text"""
        return self.translations.get(key, default if default else key)
