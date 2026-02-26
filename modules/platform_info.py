"""Platform detection constants — single source of truth."""

import platform
import sys

IS_APPLE_SILICON: bool = sys.platform == "darwin" and platform.machine() == "arm64"
IS_WINDOWS: bool = sys.platform == "win32"
IS_LINUX: bool = sys.platform == "linux"
