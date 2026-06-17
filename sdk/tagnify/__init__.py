"""
Tagnify LLM-powered automated data labeling

    pip install tagnify
"""

from tagnify.schema import Schema, Example, LabelResult
from tagnify.client import Tagnify

__all__ = ["Tagnify", "Schema", "Example", "LabelResult"]
__version__ = "0.2.0"