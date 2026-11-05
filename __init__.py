"""
ComfyUI OpenAI API Nodes

This package provides ComfyUI custom nodes for OpenAI API integration.
Includes nodes for DALL-E 2, DALL-E 3, GPT Image 1, and OpenAI Chat functionality.
"""

from .canary import NODE_CLASS_MAPPINGS as CANARY_MAPPINGS
from .nodes_openai import NODE_CLASS_MAPPINGS as OPENAI_MAPPINGS
from .nodes_openai import NODE_DISPLAY_NAME_MAPPINGS as OPENAI_DISPLAY_MAPPINGS

# Combine all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(CANARY_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OPENAI_MAPPINGS)

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(OPENAI_DISPLAY_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]