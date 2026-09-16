"""Run the comprehensive measurement stage.

Reporting remains separate so figures can be regenerated without repeating
expensive benchmark measurements.
"""

from .run_runtime_suite import main


if __name__ == "__main__":
    raise SystemExit(main())
