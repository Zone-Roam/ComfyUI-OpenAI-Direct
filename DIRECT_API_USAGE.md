# OpenAI Direct API Usage Guide

## Overview

The new **OpenAI Direct API nodes** allow you to generate and edit images by calling OpenAI's API directly using your own API key, bypassing ComfyUI's proxy service.

### Available Nodes:
1. **OpenAI GPT Image 1 (Direct API)** - Complete GPT Image 1 functionality with direct API access
2. **OpenAI Image Editor (Direct API)** - Edit existing images with DALL-E 2 and optional masks

## Benefits

- **No ComfyUI billing**: Use your own OpenAI account and pay OpenAI directly
- **Cost transparency**: See exact OpenAI pricing without markup
- **Direct control**: Full access to all OpenAI parameters
- **No account limits**: Use your own OpenAI rate limits
- **Privacy**: Direct communication with OpenAI (no proxy)

## Setup

### 1. Get OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key (starts with `sk-`)
5. Copy and save your API key securely

### 2. Add Credits to OpenAI Account

- Add credits to your OpenAI account for API usage
- Check current pricing at [OpenAI Pricing](https://openai.com/pricing)

## Using the Node

### 1. Add the Nodes

In ComfyUI, find the nodes under:
```
Add Node > api node > image > OpenAI Direct > OpenAI GPT Image 1 (Direct API)
Add Node > api node > image > OpenAI Direct > OpenAI Image Editor (Direct API)
```

### 2. Configure Inputs

#### For GPT Image 1 (Direct API):

**Required Inputs:**
- **api_key**: Your OpenAI API key (starts with `sk-`)
- **prompt**: Text description for the image you want to generate

**Optional Inputs:**
- **seed**: Seed for reproducible generation (0 = random, 1-2³¹⁻¹ = fixed) - *Currently not supported by GPT Image 1 API*
- **quality**: Image quality (`low`, `medium`, `high`) - affects cost and generation time
- **background**: Background type (`opaque`, `transparent`)
- **size**: Image dimensions (`auto`, `1024x1024`, `1024x1536`, `1536x1024`)
- **n**: Number of images to generate (1-8)
- **image**: Optional reference image for editing
- **mask**: Optional mask for inpainting (white areas will be replaced)

#### For Image Editor:

**Required Inputs:**
- **api_key**: Your OpenAI API key (starts with `sk-`)
- **image**: The original image to edit (any format, will be converted to PNG)
- **prompt**: Text description of the desired edit

**Optional Inputs:**
- **mask**: Mask image where transparent areas indicate regions to edit
  - If not provided, the entire image may be modified
  - Black areas = keep unchanged, Transparent areas = edit
- **size**: Output image size (`256x256`, `512x512`, `1024x1024`)
- **n**: Number of edited variations to generate (1-10)

### 3. Example Configurations

#### GPT Image 1 Generation Example:
```
API Key: sk-your-api-key-here
Prompt: A serene mountain landscape at sunset with a lake reflection
Seed: 0 (random)
Quality: medium
Background: opaque
Size: auto
N: 1
```

#### GPT Image 1 Editing Example:
```
API Key: sk-your-api-key-here
Image: [Upload your image]
Prompt: Add a rainbow in the sky
Quality: high
Background: opaque
Size: 1024x1024
N: 2
```

#### GPT Image 1 Inpainting Example:
```
API Key: sk-your-api-key-here
Image: [Upload your image]
Mask: [Upload mask - white areas will be replaced]
Prompt: Replace with a beautiful garden
Quality: high
Background: transparent
Size: auto
N: 1
```

#### Image Editing Example:
```
API Key: sk-your-api-key-here
Image: [Upload your image]
Prompt: Add a rainbow in the sky
Mask: [Optional - upload mask to target specific areas]
Size: 1024x1024
N: 2
```

## Cost Information

### DALL-E 3 Pricing (as of 2024):
- Standard 1024×1024: $0.040 per image
- Standard 1024×1792 or 1792×1024: $0.080 per image
- HD 1024×1024: $0.080 per image
- HD 1024×1792 or 1792×1024: $0.120 per image

### DALL-E 2 Pricing:
- **Generation**: 1024×1024: $0.020, 512×512: $0.018, 256×256: $0.016 per image
- **Editing**: 1024×1024: $0.020, 512×512: $0.018, 256×256: $0.016 per image

*Note: Image editing uses DALL-E 2 only. Prices may change. Check OpenAI's official pricing page for current rates.*

## Error Handling

The node includes comprehensive error handling:

- **Invalid API Key**: Clear error message if API key is missing or invalid
- **Network Errors**: Timeout and connection error handling
- **API Errors**: Detailed OpenAI API error messages
- **Image Download**: Automatic retry and error handling for image downloads

## Security Notes

- **Never share your API key**: Keep it secure and private
- **Use environment variables**: Consider storing API key in environment variables
- **Monitor usage**: Check your OpenAI dashboard for usage and costs
- **Set usage limits**: Configure spending limits in your OpenAI account

## Troubleshooting

### Common Issues:

1. **"Invalid API Key" Error**
   - Ensure API key starts with `sk-`
   - Check that the key is copied correctly
   - Verify the key is active in your OpenAI account

2. **"Insufficient Credits" Error**
   - Add credits to your OpenAI account
   - Check your account balance

3. **"Rate Limit" Error**
   - Wait a moment and try again
   - Check your OpenAI rate limits

4. **Network Timeout**
   - Check your internet connection
   - Try again after a moment

5. **Connection Refused (WinError 10061)**
   - This is common in China or behind corporate firewalls
   - Set up a proxy server (Clash, V2Ray, etc.)
   - Fill in the proxy_url field with your proxy address
   - Common proxy URLs: `http://127.0.0.1:7890` (Clash), `http://127.0.0.1:10809` (V2Ray)

6. **Proxy Connection Error**
   - Make sure your proxy server is running
   - Test the proxy with a web browser first
   - Verify the proxy URL is correct
   - Ensure the proxy supports HTTPS

7. **Seed Parameter Not Working**
   - The seed parameter is currently not supported by GPT Image 1 API
   - Images are generated with random seeds regardless of the seed value
   - The parameter is preserved in the UI for future compatibility
   - This is a limitation of the current OpenAI API, not the node

## Comparison: Direct API vs ComfyUI Proxy

| Feature | Direct API | ComfyUI Proxy |
|---------|------------|---------------|
| Billing | Your OpenAI account | ComfyUI account |
| Cost | OpenAI pricing | ComfyUI pricing (may include markup) |
| Setup | Need OpenAI API key | Need ComfyUI account |
| Privacy | Direct to OpenAI | Through ComfyUI proxy |
| Rate Limits | Your OpenAI limits | ComfyUI limits |
| Features | All OpenAI features | Limited by proxy |

Choose the option that best fits your needs!
