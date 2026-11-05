"""Project-wide constants (random seed, time zone, test period)."""
import pandas as pd

SEED = 42
TZ = "Europe/Paris"

# Test period: the last 12 months of the data, in Paris local days. End is exclusive.
TEST_START = pd.Timestamp("2025-09-01", tz=TZ)
TEST_END = pd.Timestamp("2026-09-01", tz=TZ)
