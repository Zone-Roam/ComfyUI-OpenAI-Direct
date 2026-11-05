# Changelog - ComfyUI OpenAI Nodes

## Version 1.6.0 - Fixed GPT Image 1 API Compatibility

### Bug Fixes:
- **Fixed seed parameter issue**: Removed unsupported seed parameter from GPT Image 1 API requests
  - Commented out seed parameter in API payload to prevent "Unknown parameter: 'seed'" error
  - Preserved seed parameter in UI for future compatibility
  - Updated tooltip to explain current API limitation
  - Added explanatory comments in code
- **Fixed parameter definition errors**: Ensured all proxy_url parameters are properly defined
  - Added missing proxy_url parameter to method signatures
  - Fixed parameter placement in INPUT_TYPES definitions

### Technical Changes:
- Commented out seed parameter transmission in both generation and editing endpoints
- Maintained UI compatibility for future API updates
- Enhanced error handling and user feedback

## Version 1.5.0 - Added Automatic Proxy Detection

### New Features:
- **Automatic Proxy Detection**: Smart proxy detection that works without manual configuration
  - Automatically detects system proxy settings
  - Tests common proxy ports (7890, 10809, 1080, 8080)
  - Works with Clash, V2Ray, and other popular proxy software
  - Falls back to direct connection if no proxy found
  - User-specified proxy_url still takes highest priority
- **Enhanced Proxy Support**: Improved proxy handling for all network requests
  - Proxy support for API calls and image downloads
  - Detailed logging of proxy detection and usage
  - Graceful fallback mechanisms

### Technical Changes:
- Added `get_system_proxy()` function for automatic detection
- Enhanced proxy setup logic in both Direct API nodes
- Added urllib.request for system proxy detection
- Improved proxy testing with quick connectivity checks
- Better user feedback with detailed proxy status messages

## Version 1.3.0 - Added GPT Image 1 Direct API Support

### New Features:
- **OpenAI GPT Image 1 (Direct API)**: Complete replica of original node with direct API access
  - Identical interface to original OpenAI GPT Image 1 node
  - Added API key input field for direct OpenAI authentication
  - Supports all original features: generation, editing, inpainting
  - Preserves all original parameters: seed, quality, background, size, n
  - Direct API calls to OpenAI's GPT Image 1 endpoint
  - Category: `api node/image/OpenAI Direct`

### Technical Changes:
- Enhanced image processing with transparency support
- Automatic endpoint selection (generation vs editing)
- Multipart form data handling for image editing
- Proper mask handling for inpainting operations
- RGBA support for transparent backgrounds

## Version 1.2.0 - Added Image Editing Support

### New Features:
- **OpenAI Image Editor (Direct API)**: New node for editing existing images
  - Uses DALL-E 2 for image editing via direct API calls
  - Supports image upload and optional mask for targeted editing
  - Automatic image format conversion (PNG, square format)
  - Multiple output sizes (256x256, 512x512, 1024x1024)
  - Batch editing (up to 10 variations)
  - Category: `api node/image/OpenAI Direct`

### Technical Changes:
- New class `OpenAIImageEditDirect` in `nodes_openai.py`
- Added image processing utilities (`tensor_to_pil`, `pil_to_png_bytes`)
- Automatic image cropping to square format
- Support for multipart/form-data uploads
- Enhanced error handling for image processing

## Version 1.1.0 - Added Direct API Support

### New Features:
- **OpenAI Image Generator (Direct API)**: New node that calls OpenAI API directly
  - Uses your own OpenAI API key (no ComfyUI billing)
  - Supports DALL-E 2 and DALL-E 3
  - Full parameter control (size, quality, style, batch generation)
  - Direct cost transparency with OpenAI
  - Category: `api node/image/OpenAI Direct`

### Technical Changes:
- Added `requests` as a required dependency
- Added `base64` import for image processing
- New class `OpenAIGPTImage1Direct` in `nodes_openai.py`
- Updated node mappings and display names

## Version 1.0.0 - Simplified OpenAI-Only Version

### What was removed:
- All non-OpenAI API nodes (BFL, Luma, Kling, Pika, Pixverse, Recraft, Rodin, Runway, Stability, Tripo, VEO2, Gemini, Ideogram, MiniMax, MoonValley)
- Corresponding API client files for removed services
- Development configuration files (redocly configs)
- Unused dependencies and imports

### What was kept:
- **OpenAI DALL·E 2** - Image generation and editing with masks
- **OpenAI DALL·E 3** - Advanced image generation with style controls
- **OpenAI GPT Image 1** - Latest image generation model
- **OpenAI Chat** - Text generation and multi-turn conversations
- **OpenAI Chat Input Files** - File input support for chat
- **OpenAI Chat Advanced Options** - Configuration for chat parameters

### Core Dependencies Retained:
- `apis/client.py` - Core API client framework
- `apis/request_logger.py` - Request logging functionality
- `apinode_utils.py` - Utility functions for API nodes
- `mapper_utils.py` - Model field mapping utilities
- `util/validation_utils.py` - Validation utilities
- `canary.py` - Version compatibility checks

### Supported OpenAI Models:
- GPT-4 series (gpt-4, gpt-4-turbo, gpt-4o, etc.)
- GPT-3.5 series
- O1 series (o1, o1-mini, o1-pro)
- O3 series (o3-mini)
- O4 series (o4-mini)
- DALL-E 2 and DALL-E 3
- GPT Image 1

### Features:
- **Image Generation**: Text-to-image, image editing, style controls
- **Text Generation**: Multi-turn conversations, file inputs, advanced options
- **Authentication**: Comfy.org API integration
- **Error Handling**: Comprehensive error handling and validation

### File Structure:
```
comfyui-openai-nodes/
├── __init__.py                 # Main package init
├── README.md                   # Documentation
├── requirements.txt            # Dependencies
├── CHANGELOG.md               # This file
├── test_structure.py          # Structure validation script
├── nodes_openai.py            # OpenAI node implementations
├── canary.py                  # Compatibility checks
├── apinode_utils.py           # API utilities
├── mapper_utils.py            # Field mapping utilities
├── apis/
│   ├── __init__.py            # OpenAI API models
│   ├── client.py              # Core API client
│   └── request_logger.py      # Request logging
└── util/
    ├── __init__.py
    └── validation_utils.py    # Validation utilities
```

This simplified version focuses exclusively on OpenAI functionality while maintaining all the core features and capabilities of the original OpenAI nodes.
