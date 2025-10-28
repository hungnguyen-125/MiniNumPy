import sys
from pathlib import Path

# Add the src folder to the Python import path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from MiniNumPy import *