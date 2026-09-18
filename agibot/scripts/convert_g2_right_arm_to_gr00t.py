#!/usr/bin/env python3
"""Generic entry point for the validated G2 right-arm GR00T converter.

Use ``--episode-manifest`` to keep task-specific train and held-out datasets
strictly separated. The implementation remains in the legacy grasp-named
module to preserve compatibility with existing commands.
"""

from convert_xichong_right_single_grasp import main


if __name__ == "__main__":
    raise SystemExit(main())
