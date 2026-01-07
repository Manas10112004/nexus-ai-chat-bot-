import sys
import os

# --- FIX: Add the parent directory to the system path ---
# This tells Python: "Look in the folder above this one for nexus_engine.py"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from io import BytesIO
# Now this import will work because Python knows where to look
from nexus_engine import DataEngine


# Fixture to initialize the engine before each test
@pytest.fixture
def engine():
    return DataEngine()


def test_initialization(engine):
    """Test that the engine starts with a clean state."""
    assert engine.df is None
    assert "pd" in engine.scope
    assert "plt" in engine.scope


def test_load_csv(engine):
    """Test loading a valid CSV file."""
    # Create a dummy CSV in memory
    csv_content = b"col1,col2\n1,10\n2,20\n3,30"
    dummy_file = BytesIO(csv_content)
    dummy_file.name = "test_data.csv"

    # Load it
    status = engine.load_file(dummy_file)

    # Assertions
    assert "Data Loaded" in status
    assert engine.df is not None
    assert len(engine.df) == 3
    assert "col1" in engine.df.columns