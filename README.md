# ComfyUI OpenAI Direct API Nodes

> **Attribution.** This package is a trimmed and modified copy of the OpenAI nodes from ComfyUI's official API nodes ([comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI), `comfy_api_nodes`), licensed under GPL-3.0. It stays under GPL-3.0 (see `LICENSE`).
>
> **What I added:** "Direct API" versions of the GPT Image 1 and image-editor nodes that call OpenAI with your own API key instead of ComfyUI account credits, plus proxy support with automatic detection. Built in 2025 for a batch image and video production pipeline.

## Introduction

This is a simplified version of ComfyUI API Nodes that contains only OpenAI-related nodes. These nodes work by calling OpenAI's external APIs.

## Available Nodes

### ComfyUI Proxy Nodes (Original)
- **OpenAI DALL·E 2**: Generate images using DALL·E 2 model
- **OpenAI DALL·E 3**: Generate images using DALL·E 3 model
- **OpenAI GPT Image 1**: Generate images using GPT Image 1 model
- **OpenAI Chat**: Text generation and chat functionality with OpenAI models
- **OpenAI Chat Input Files**: Load files as input for OpenAI Chat
- **OpenAI Chat Advanced Options**: Configure advanced options for OpenAI Chat

### Direct API Nodes (New)
- **OpenAI GPT Image 1 (Direct API)**: GPT Image 1 with direct API access
  - Identical interface to original node + API key input
  - Supports generation, editing, and inpainting
  - All original features: quality, background, size, seed controls
  - No ComfyUI account billing required
- **OpenAI Image Editor (Direct API)**: Edit existing images using DALL-E 2
  - Upload image and optional mask for targeted editing
  - Direct API calls to OpenAI
  - Supports various output sizes

## Supported OpenAI Models

- GPT-4 series (gpt-4, gpt-4-turbo, gpt-4o, etc.)
- GPT-3.5 series
- O1 series (o1, o1-mini, o1-pro)
- O3 series (o3-mini)
- O4 series (o4-mini)
- DALL-E 2 and DALL-E 3
- GPT Image 1

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone <repository-url> comfyui-openai-nodes
```

2. Install dependencies:
```bash
cd comfyui-openai-nodes
pip install -r requirements.txt
```

3. Restart ComfyUI

## Usage

After installation, you'll find the OpenAI nodes in the "api node" category in ComfyUI:
- `api node/image/OpenAI` - Original image generation nodes (via ComfyUI proxy)
- `api node/image/OpenAI Direct` - Direct API image generation nodes
- `api node/text/OpenAI` - Text and chat nodes

## Authentication

### For Original Nodes (ComfyUI Proxy)
These nodes require authentication with the Comfy.org API service. You'll need:
- An auth token from Comfy.org
- A Comfy API key

### For Direct API Nodes (New)
These nodes call OpenAI directly and require:
- Your own OpenAI API key (starts with 'sk-')
- Automatic proxy detection (works with Clash, V2Ray, etc.)
- Optional manual proxy URL for custom setups
- No ComfyUI account or billing required

## Dependencies

This simplified version only includes the core dependencies needed for OpenAI functionality:
- `pydantic` - For API model validation
- `torch` - For tensor operations
- `PIL` - For image processing
- `numpy` - For numerical operations

## Features

### Image Generation (Original Nodes)
- **DALL·E 2**: Supports text-to-image and image editing with masks
- **DALL·E 3**: Advanced text-to-image with style and quality controls
- **GPT Image 1**: Latest image generation model with background control

### GPT Image 1 (Direct API)
- **Identical interface**: Same as original node with API key input added
- **Complete feature parity**: Generation, editing, inpainting, transparency
- **All original controls**: Seed, quality (low/medium/high), background, size
- **Automatic proxy detection**: Works with Clash, V2Ray out of the box
- **Smart network handling**: Auto-detects proxy or uses direct connection
- **Enhanced error handling**: Detailed error messages with troubleshooting tips
- **Direct OpenAI API calls**: No ComfyUI proxy, direct communication
- **Cost transparency**: Pay OpenAI directly, no markup

### Image Editing (Direct API)
- **DALL-E 2 image editing**: Edit existing images with text prompts
- **Mask support**: Optional mask for targeted editing (transparent areas are edited)
- **Flexible input**: Upload any image, automatically converted to required format
- **Multiple outputs**: Generate up to 10 edited variations
- **Square format**: Automatically crops images to square format as required by API

### Text Generation
- **Multi-turn conversations**: Persistent chat context
- **File inputs**: Support for PDF and text files
- **Advanced options**: Temperature, max tokens, and other parameters
- **Multiple models**: Support for all OpenAI text models

## Original Project

This is a simplified version of the full ComfyUI API Nodes project. The original project includes many more API integrations and can be found at the official ComfyUI API Nodes repository.

## License

This project follows the same license as the original ComfyUI API Nodes project.