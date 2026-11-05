import io
from typing import TypedDict, Optional
import json
import os
import time
import re
import uuid
import requests
import base64
import urllib.request
from enum import Enum
from inspect import cleandoc
import numpy as np
import torch
from PIL import Image
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from server import PromptServer
import folder_paths
import traceback
import random
from io import BytesIO


def get_system_proxy():
    """
    Try to detect system proxy settings automatically.
    Returns a proxy dict if found, empty dict otherwise.
    """
    try:
        # Try to get system proxy settings
        proxy_handler = urllib.request.ProxyHandler()
        proxy_dict = proxy_handler.proxies

        # Check for http/https proxy
        if 'http' in proxy_dict or 'https' in proxy_dict:
            http_proxy = proxy_dict.get('http') or proxy_dict.get('https')
            if http_proxy:
                print(f"Detected system proxy: {http_proxy}")
                return {
                    'http': http_proxy,
                    'https': http_proxy
                }

        # Try common proxy ports if no system proxy found
        common_proxies = [
            'http://127.0.0.1:7890',  # Clash
            'http://127.0.0.1:10809', # V2Ray
            'http://127.0.0.1:1080',  # SOCKS proxy converted
            'http://127.0.0.1:8080',  # Common HTTP proxy
        ]

        for proxy_url in common_proxies:
            try:
                # Quick test to see if proxy is responding
                test_response = requests.get(
                    'http://httpbin.org/ip',
                    proxies={'http': proxy_url, 'https': proxy_url},
                    timeout=3
                )
                if test_response.status_code == 200:
                    print(f"Auto-detected working proxy: {proxy_url}")
                    return {
                        'http': proxy_url,
                        'https': proxy_url
                    }
            except:
                continue

    except Exception as e:
        print(f"Could not detect system proxy: {e}")

    return {}


from comfy_api_nodes.apis import (
    OpenAIImageGenerationRequest,
    OpenAIImageEditRequest,
    OpenAIImageGenerationResponse,
    OpenAICreateResponse,
    OpenAIResponse,
    CreateModelResponseProperties,
    Item,
    Includable,
    OutputContent,
    InputImageContent,
    Detail,
    InputTextContent,
    InputMessage,
    InputMessageContentList,
    InputContent,
    InputFileContent,
)

from comfy_api_nodes.apis.client import (
    ApiEndpoint,
    HttpMethod,
    SynchronousOperation,
    PollingOperation,
    EmptyRequest,
)

from comfy_api_nodes.apinode_utils import (
    downscale_image_tensor,
    validate_and_cast_response,
    validate_string,
    tensor_to_base64_string,
    text_filepath_to_data_uri,
)
from comfy_api_nodes.mapper_utils import model_field_to_node_input


RESPONSES_ENDPOINT = "/proxy/openai/v1/responses"
STARTING_POINT_ID_PATTERN = r"<starting_point_id:(.*)>"


class HistoryEntry(TypedDict):
    """Type definition for a single history entry in the chat."""

    prompt: str
    response: str
    response_id: str
    timestamp: float


class ChatHistory(TypedDict):
    """Type definition for the chat history dictionary."""

    __annotations__: dict[str, list[HistoryEntry]]


class SupportedOpenAIModel(str, Enum):
    o4_mini = "o4-mini"
    o1 = "o1"
    o3 = "o3"
    o1_pro = "o1-pro"
    gpt_4o = "gpt-4o"
    gpt_4_1 = "gpt-4.1"
    gpt_4_1_mini = "gpt-4.1-mini"
    gpt_4_1_nano = "gpt-4.1-nano"


class OpenAIDalle2(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 2 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for DALL·E",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["256x256", "512x512", "1024x1024"],
                        "default": "1024x1024",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(
        self,
        prompt,
        seed=0,
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-2"
        path = "/proxy/openai/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        img_binary = None

        if image is not None and mask is not None:
            path = "/proxy/openai/images/edits"
            content_type = "multipart/form-data"
            request_class = OpenAIImageEditRequest

            input_tensor = image.squeeze().cpu()
            height, width, channels = input_tensor.shape
            rgba_tensor = torch.ones(height, width, 4, device="cpu")
            rgba_tensor[:, :, :channels] = input_tensor

            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            rgba_tensor[:, :, 3] = 1 - mask.squeeze().cpu()

            rgba_tensor = downscale_image_tensor(rgba_tensor.unsqueeze(0)).squeeze()

            image_np = (rgba_tensor.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_binary = img_byte_arr  # .getvalue()
            img_binary.name = "image.png"
        elif image is not None or mask is not None:
            raise Exception("Dall-E 2 image editing requires an image AND a mask")

        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                seed=seed,
            ),
            files=(
                {
                    "image": img_binary,
                }
                if img_binary
                else None
            ),
            content_type=content_type,
            auth_kwargs=kwargs,
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class OpenAIDalle3(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's DALL·E 3 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for DALL·E",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["standard", "hd"],
                        "default": "standard",
                        "tooltip": "Image quality",
                    },
                ),
                "style": (
                    IO.COMBO,
                    {
                        "options": ["natural", "vivid"],
                        "default": "natural",
                        "tooltip": "Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["1024x1024", "1024x1792", "1792x1024"],
                        "default": "1024x1024",
                        "tooltip": "Image size",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(
        self,
        prompt,
        seed=0,
        style="natural",
        quality="standard",
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "dall-e-3"

        # build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path="/proxy/openai/images/generations",
                method=HttpMethod.POST,
                request_model=OpenAIImageGenerationRequest,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=OpenAIImageGenerationRequest(
                model=model,
                prompt=prompt,
                quality=quality,
                size=size,
                style=style,
                seed=seed,
            ),
            auth_kwargs=kwargs,
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class OpenAIGPTImage1(ComfyNodeABC):
    """
    Generates images synchronously via OpenAI's GPT Image 1 endpoint.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for GPT Image 1",
                    },
                ),
            },
            "optional": {
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["low", "medium", "high"],
                        "default": "low",
                        "tooltip": "Image quality, affects cost and generation time.",
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": "Return image with or without background",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                        "default": "auto",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI"
    DESCRIPTION = cleandoc(__doc__ or "")
    API_NODE = True

    def api_call(
        self,
        prompt,
        seed=0,
        quality="low",
        background="opaque",
        image=None,
        mask=None,
        n=1,
        size="1024x1024",
        unique_id=None,
        **kwargs,
    ):
        validate_string(prompt, strip_whitespace=False)
        model = "gpt-image-1"
        path = "/proxy/openai/images/generations"
        content_type = "application/json"
        request_class = OpenAIImageGenerationRequest
        img_binaries = []
        mask_binary = None
        files = []

        if image is not None:
            path = "/proxy/openai/images/edits"
            request_class = OpenAIImageEditRequest
            content_type = "multipart/form-data"

            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_image_tensor(single_image).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)
                img_binary = img_byte_arr
                img_binary.name = f"image_{i}.png"

                img_binaries.append(img_binary)
                if batch_size == 1:
                    files.append(("image", img_binary))
                else:
                    files.append(("image[]", img_binary))

        if mask is not None:
            if image is None:
                raise Exception("Cannot use a mask without an input image")
            if image.shape[0] != 1:
                raise Exception("Cannot use a mask with multiple image")
            if mask.shape[1:] != image.shape[1:-1]:
                raise Exception("Mask and Image must be the same size")
            batch, height, width = mask.shape
            rgba_mask = torch.zeros(height, width, 4, device="cpu")
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

            scaled_mask = downscale_image_tensor(rgba_mask.unsqueeze(0)).squeeze()

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_np)
            mask_img_byte_arr = io.BytesIO()
            mask_img.save(mask_img_byte_arr, format="PNG")
            mask_img_byte_arr.seek(0)
            mask_binary = mask_img_byte_arr
            mask_binary.name = "mask.png"
            files.append(("mask", mask_binary))

        # Build the operation
        operation = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=path,
                method=HttpMethod.POST,
                request_model=request_class,
                response_model=OpenAIImageGenerationResponse,
            ),
            request=request_class(
                model=model,
                prompt=prompt,
                quality=quality,
                background=background,
                n=n,
                seed=seed,
                size=size,
            ),
            files=files if files else None,
            content_type=content_type,
            auth_kwargs=kwargs,
        )

        response = operation.execute()

        img_tensor = validate_and_cast_response(response, node_id=unique_id)
        return (img_tensor,)


class OpenAIGPTImage1Direct(ComfyNodeABC):
    """
    Generates images directly via OpenAI's GPT-4o API using your own API key.
    This uses the chat completions API with vision capabilities for image generation.
    """

    def __init__(self):
        # 获取节点所在目录
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.node_dir, "gpt_api_config.json")
        self.log_messages = []

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        # 创建临时实例以加载配置
        temp_instance = cls()
        config = temp_instance.load_config()

        return {
            "required": {
                "api_key": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "Your OpenAI API Key (sk-...)",
                    },
                ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for image generation",
                    },
                ),
            },
            "optional": {
                "proxy_url": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "Optional proxy URL (e.g., http://127.0.0.1:7890) - leave empty for auto-detection",
                    },
                ),
                "model": (
                    IO.COMBO,
                    {
                        "options": ["gpt-image-1", "dall-e-3", "dall-e-2"],
                        "default": "gpt-image-1",
                        "tooltip": "Image generation model",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["1024x1024", "1024x1792", "1792x1024"],
                        "default": "1024x1024",
                        "tooltip": "Image size (dall-e-3 only supports these sizes)",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["standard", "hd"],
                        "default": "standard",
                        "tooltip": "Image quality (dall-e-3 only)",
                    },
                ),
                "style": (
                    IO.COMBO,
                    {
                        "options": ["natural", "vivid"],
                        "default": "natural",
                        "tooltip": "Image style (dall-e-3 only)",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Number of images to generate",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "generate_image"
    CATEGORY = "api node/image/OpenAI Direct"
    DESCRIPTION = cleandoc(__doc__ or "")

    def log(self, message):
        """全局日志函数：记录到日志列表"""
        if hasattr(self, 'log_messages'):
            self.log_messages.append(message)
        print(f"[OpenAI GPT Image] {message}")
        return message

    def save_config(self, api_key, api_url, model):
        """保存API配置到文件"""
        if not api_key or len(api_key) < 10:
            return False
        try:
            config = {
                "api_key": api_key,
                "api_url": api_url,
                "model": model,
                "saved_time": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(self.config_file, "w") as f:
                json.dump(config, f)
            self.log("已保存API配置到节点目录")
            return True
        except Exception as e:
            self.log(f"保存API配置失败: {e}")
            return False

    def load_config(self):
        """从文件加载API配置"""
        default_config = {
            "api_key": "",
            "api_url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-4o"
        }
        if not os.path.exists(self.config_file):
            return default_config
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            # 验证必要字段
            if "api_key" not in config or len(config["api_key"]) < 10:
                self.log("已保存的API密钥无效")
                return default_config
            self.log("成功加载已保存的API配置")
            return config
        except Exception as e:
            self.log(f"加载API配置失败: {e}")
            return default_config



    def generate_image(
        self,
        api_key: str,
        prompt: str,
        proxy_url: str = "",
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
        n: int = 1,
    ):
        """使用DALL-E API生成图像"""
        self.log_messages = []  # 重置日志

        try:
            # 验证输入
            if not api_key or not api_key.startswith("sk-"):
                raise ValueError("请提供有效的OpenAI API密钥 (以'sk-'开头)")

            if not prompt.strip():
                raise ValueError("请提供图像生成的提示词")

            # 设置代理
            proxies = {}
            if proxy_url.strip():
                proxies = {
                    'http': proxy_url.strip(),
                    'https': proxy_url.strip()
                }
                self.log(f"使用用户指定的代理: {proxy_url.strip()}")
            else:
                proxies = get_system_proxy()
                if proxies:
                    self.log(f"使用自动检测的代理: {proxies.get('http', 'Unknown')}")
                else:
                    self.log("未配置代理，尝试直接连接")

            # 构建API请求
            url = "https://api.openai.com/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": model,
                "prompt": prompt,
                "n": n,
                "size": size
            }

            # 根据模型添加特定参数
            if model == "gpt-image-1":
                # GPT Image 1 可能支持的参数
                payload["quality"] = quality
                payload["style"] = style
            elif model == "dall-e-3":
                # DALL-E 3特有的参数
                payload["quality"] = quality
                payload["style"] = style

            self.log(f"发送请求到: {url}")
            self.log(f"使用模型: {model}")
            self.log(f"图像尺寸: {size}")
            self.log(f"生成数量: {n}")
            if model in ["gpt-image-1", "dall-e-3"]:
                self.log(f"质量: {quality}, 风格: {style}")

            # 发送API请求
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                proxies=proxies,
                timeout=120
            )

            self.log(f"API响应状态: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"OpenAI API错误 ({response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error'].get('message', '未知错误')}"
                except:
                    error_msg += f": {response.text}"
                self.log(f"API错误: {error_msg}")
                raise Exception(error_msg)

            # 解析响应
            result = response.json()

            if "data" not in result or not result["data"]:
                error_msg = "API响应中没有data字段或为空"
                self.log(error_msg)
                raise Exception(error_msg)

            # 下载并处理图像
            images = []
            self.log(f"API返回 {len(result['data'])} 张图像")

            for i, img_data in enumerate(result["data"]):
                if "url" not in img_data:
                    self.log(f"图像 {i}: 响应数据中没有URL")
                    continue

                img_url = img_data["url"]
                self.log(f"下载图像 {i} 从: {img_url[:50]}...")

                try:
                    img_response = requests.get(img_url, proxies=proxies, timeout=30)
                    self.log(f"图像 {i} 下载响应: {img_response.status_code}")

                    if img_response.status_code == 200:
                        # 转换为PIL图像
                        img = Image.open(io.BytesIO(img_response.content))

                        # 转换为RGB
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # 转换为张量
                        img_array = np.array(img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        images.append(img_tensor)
                        self.log(f"成功处理图像 {i}: {img.size}")

                    else:
                        self.log(f"下载图像 {i} 失败: HTTP {img_response.status_code}")

                except Exception as download_error:
                    self.log(f"下载图像 {i} 时出错: {download_error}")

            self.log(f"成功下载 {len(images)} / {len(result['data'])} 张图像")

            if not images:
                raise Exception("未能下载任何图像")

            # 合并所有图像为批次
            result_tensor = torch.cat(images, dim=0)
            self.log("图像生成完成")
            return (result_tensor,)

        except Exception as e:
            error_msg = f"生成图像时出错: {str(e)}"
            self.log(error_msg)
            self.log(f"错误详情: {traceback.format_exc()}")
            raise Exception(error_msg)


class OpenAIGPTImage1Direct_Simple(ComfyNodeABC):
    """
    GPT Image 1 Direct - 简单版本，基于官方节点，只需输入API key
    支持文生图和图像编辑功能
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "api_key": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "Your OpenAI API Key (sk-...)",
                    },
                ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt for GPT Image 1",
                    },
                ),
            },
            "optional": {
                "proxy_url": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "Optional proxy URL (e.g., http://127.0.0.1:7890)",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2**31 - 1,
                        "step": 1,
                        "display": "number",
                        "control_after_generate": True,
                        "tooltip": "not implemented yet in backend",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["low", "medium", "high"],
                        "default": "low",
                        "tooltip": "Image quality, affects cost and generation time.",
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": "Return image with or without background",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                        "default": "auto",
                        "tooltip": "Image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "display": "number",
                        "tooltip": "How many images to generate",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional reference image for image editing.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "default": None,
                        "tooltip": "Optional mask for inpainting (white areas will be replaced)",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api node/image/OpenAI Direct"
    DESCRIPTION = cleandoc(__doc__ or "")

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image."""
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        if tensor.max() <= 1.0:
            tensor = tensor * 255.0

        tensor = tensor.clamp(0, 255).byte()

        if tensor.shape[0] == 3:  # CHW format
            array = tensor.permute(1, 2, 0).cpu().numpy()
        else:  # HWC format
            array = tensor.cpu().numpy()

        return Image.fromarray(array)

    def pil_to_png_bytes(self, pil_image):
        """Convert PIL image to PNG bytes."""
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    def test_proxy_connection(self, proxies):
        """测试代理连接是否正常"""
        try:
            test_url = "https://httpbin.org/ip"
            print(f"[GPT Image 1] Testing proxy connection to: {test_url}")
            response = requests.get(test_url, proxies=proxies, timeout=10)
            if response.status_code == 200:
                ip_info = response.json()
                print(f"[GPT Image 1] Proxy test successful, IP: {ip_info.get('origin', 'Unknown')}")
                return True
            else:
                print(f"[GPT Image 1] Proxy test failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"[GPT Image 1] Proxy test error: {e}")
            return False

    def api_call(
        self,
        api_key: str,
        prompt: str,
        proxy_url: str = "",
        seed: int = 0,
        quality: str = "low",
        background: str = "opaque",
        image=None,
        mask=None,
        n: int = 1,
        size: str = "auto",
    ):
        """直接调用OpenAI GPT Image 1 API"""
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Please provide a valid OpenAI API key (starts with 'sk-')")

        if not prompt.strip():
            raise ValueError("Please provide a prompt for image generation")

        try:
            # Setup proxy
            proxies = {}
            if proxy_url.strip():
                proxies = {
                    'http': proxy_url.strip(),
                    'https': proxy_url.strip()
                }
                print(f"[GPT Image 1] Using user-specified proxy: {proxy_url.strip()}")
                print(f"[GPT Image 1] Proxy config: {proxies}")
            else:
                proxies = get_system_proxy()
                if proxies:
                    print(f"[GPT Image 1] Using auto-detected proxy: {proxies.get('http', 'Unknown')}")
                    print(f"[GPT Image 1] Auto-detected proxy config: {proxies}")
                else:
                    print("[GPT Image 1] No proxy configured, attempting direct connection")
                    print(f"[GPT Image 1] Direct connection config: {proxies}")

            # Test proxy connection
            if proxies:
                print("[GPT Image 1] Testing proxy connection...")
                proxy_works = self.test_proxy_connection(proxies)
                if not proxy_works:
                    print("[GPT Image 1] WARNING: Proxy test failed, but continuing anyway...")

            # Determine if this is image editing or generation
            is_edit = image is not None

            if is_edit:
                # Image editing endpoint
                url = "https://api.openai.com/v1/images/edits"
                print("[GPT Image 1] Using image editing mode")

                # Convert image to PNG bytes
                pil_image = self.tensor_to_pil(image)
                image_bytes = self.pil_to_png_bytes(pil_image)

                # Prepare multipart form data
                files = {
                    'image': ('image.png', image_bytes, 'image/png'),
                    'model': (None, 'gpt-image-1'),
                    'prompt': (None, prompt),
                    'n': (None, str(n)),
                    'quality': (None, quality),
                    'background': (None, background),
                }

                if size != "auto":
                    files['size'] = (None, size)

                # Handle mask if provided
                if mask is not None:
                    # Convert mask to RGBA image with transparency
                    if len(mask.shape) == 3:
                        mask = mask[0]  # Remove batch dimension

                    height, width = mask.shape
                    rgba_mask = np.zeros((height, width, 4), dtype=np.uint8)
                    rgba_mask[:, :, 3] = (1 - mask.cpu().numpy()) * 255  # Invert mask for transparency

                    mask_img = Image.fromarray(rgba_mask, 'RGBA')
                    mask_bytes = self.pil_to_png_bytes(mask_img)
                    files['mask'] = ('mask.png', mask_bytes, 'image/png')

                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.post(url, headers=headers, files=files, proxies=proxies, timeout=60)

            else:
                # Image generation endpoint
                url = "https://api.openai.com/v1/images/generations"
                print("[GPT Image 1] Using image generation mode")

                payload = {
                    "model": "gpt-image-1",
                    "prompt": prompt,
                    "n": n,
                    "quality": quality,
                    "background": background,
                    "response_format": "b64_json"  # 请求base64格式
                }

                if size != "auto":
                    payload["size"] = size

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                response = requests.post(url, headers=headers, json=payload, proxies=proxies, timeout=60)

            print(f"[GPT Image 1] API response status: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"OpenAI API error ({response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)

            result = response.json()
            print(f"[GPT Image 1] API response content: {result}")

            # 检查不同的响应格式
            images_data = None
            if "data" in result and result["data"]:
                # 标准DALL-E格式
                images_data = result["data"]
                print(f"[GPT Image 1] Using standard DALL-E response format")
            elif "images" in result and result["images"]:
                # 可能的GPT Image 1格式
                images_data = result["images"]
                print(f"[GPT Image 1] Using GPT Image 1 response format (images field)")
            elif "url" in result:
                # 单个图像URL格式
                images_data = [result]
                print(f"[GPT Image 1] Using single image URL format")
            elif "image_url" in result:
                # 另一种单个图像格式
                images_data = [{"url": result["image_url"]}]
                print(f"[GPT Image 1] Using single image_url format")
            else:
                print(f"[GPT Image 1] Unknown response format. Available keys: {list(result.keys())}")
                print(f"[GPT Image 1] Full response: {result}")

                # 检查是否是聊天API响应（有usage字段）
                if "usage" in result:
                    print(f"[GPT Image 1] This looks like a chat API response, not image generation")
                    print(f"[GPT Image 1] You might be using the wrong API endpoint or model")
                    raise Exception("Received chat API response instead of image generation response. Check your API endpoint and model.")

                raise Exception("No images found in API response - unknown response format")

            if not images_data:
                raise Exception("No images returned from OpenAI API")

            # Download and process images
            images = []
            print(f"[GPT Image 1] API returned {len(images_data)} image(s)")

            for i, img_data in enumerate(images_data):
                print(f"[GPT Image 1] Processing image {i}, data keys: {list(img_data.keys())}")

                # GPT Image 1 返回base64编码的图像，不是URL
                if "b64_json" in img_data:
                    # Base64格式
                    print(f"[GPT Image 1] Image {i}: Found base64 data")
                    try:
                        import base64
                        base64_data = img_data["b64_json"]
                        image_data = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(image_data))
                        print(f"[GPT Image 1] Image {i} loaded from base64: {img.size}, mode: {img.mode}")

                        # Convert to RGB if necessary (preserve transparency for background=transparent)
                        if background == "transparent" and img.mode == "RGBA":
                            # Keep RGBA for transparent background
                            img_array = np.array(img).astype(np.float32) / 255.0
                            print(f"[GPT Image 1] Image {i} kept as RGBA for transparent background")
                        else:
                            # Convert to RGB for opaque background
                            if img.mode != "RGB":
                                img = img.convert("RGB")
                                print(f"[GPT Image 1] Image {i} converted to RGB")
                            img_array = np.array(img).astype(np.float32) / 255.0

                        # Convert to tensor
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        images.append(img_tensor)
                        print(f"[GPT Image 1] Successfully processed image {i}: {img.size}, tensor shape: {img_tensor.shape}")

                    except Exception as base64_error:
                        print(f"[GPT Image 1] Error processing base64 image {i}: {base64_error}")
                        import traceback
                        print(f"[GPT Image 1] Base64 error traceback: {traceback.format_exc()}")

                elif "url" in img_data:
                    # URL格式（DALL-E风格）
                    img_url = img_data["url"]
                    print(f"[GPT Image 1] Image {i}: Found URL, downloading from: {img_url[:50]}...")

                    try:
                        # 使用相同的代理配置下载图像
                        print(f"[GPT Image 1] Downloading with proxies: {proxies}")
                        img_response = requests.get(img_url, proxies=proxies, timeout=30)
                        print(f"[GPT Image 1] Image {i} download response: {img_response.status_code}")

                        if img_response.status_code == 200:
                            # Convert to PIL Image
                            img = Image.open(io.BytesIO(img_response.content))
                            print(f"[GPT Image 1] Image {i} loaded successfully: {img.size}, mode: {img.mode}")

                            # Convert to RGB if necessary (preserve transparency for background=transparent)
                            if background == "transparent" and img.mode == "RGBA":
                                # Keep RGBA for transparent background
                                img_array = np.array(img).astype(np.float32) / 255.0
                                print(f"[GPT Image 1] Image {i} kept as RGBA for transparent background")
                            else:
                                # Convert to RGB for opaque background
                                if img.mode != "RGB":
                                    img = img.convert("RGB")
                                    print(f"[GPT Image 1] Image {i} converted to RGB")
                                img_array = np.array(img).astype(np.float32) / 255.0

                            # Convert to tensor
                            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            images.append(img_tensor)
                            print(f"[GPT Image 1] Successfully processed image {i}: {img.size}, tensor shape: {img_tensor.shape}")

                        else:
                            print(f"[GPT Image 1] Failed to download image {i}: HTTP {img_response.status_code}")
                            print(f"[GPT Image 1] Response content: {img_response.text[:200]}")

                    except Exception as download_error:
                        print(f"[GPT Image 1] Error downloading image {i}: {download_error}")
                        import traceback
                        print(f"[GPT Image 1] Download error traceback: {traceback.format_exc()}")

                else:
                    print(f"[GPT Image 1] Image {i}: No URL or base64 data found")
                    print(f"[GPT Image 1] Available keys: {list(img_data.keys())}")
                    print(f"[GPT Image 1] Image data: {img_data}")

            print(f"[GPT Image 1] Successfully downloaded {len(images)} out of {len(images_data)} images")

            if not images:
                raise Exception("Failed to download any images from OpenAI")

            # Concatenate all images into a batch
            result_tensor = torch.cat(images, dim=0)
            print(f"[GPT Image 1] Generation complete, returned {len(images)} images")
            return (result_tensor,)

        except Exception as e:
            raise Exception(f"Error calling GPT Image 1 API: {str(e)}")


class OpenAIImageEditDirect(ComfyNodeABC):
    """
    Edit images directly via OpenAI's DALL-E 2 API using your own API key.
    Supports image editing with optional mask for targeted modifications.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "api_key": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "Your OpenAI API Key (sk-...)",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "tooltip": "Original image to edit (will be converted to PNG)",
                    },
                ),
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt describing the desired edit",
                    },
                ),
            },
            "optional": {
                "proxy_url": (
                    IO.STRING,
                    {
                        "default": "",
                        "tooltip": "Optional proxy URL (e.g., http://127.0.0.1:7890) - leave empty for auto-detection",
                    },
                ),
                "mask": (
                    IO.IMAGE,
                    {
                        "tooltip": "Mask image (transparent areas will be edited). If not provided, entire image may be modified.",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": ["256x256", "512x512", "1024x1024"],
                        "default": "1024x1024",
                        "tooltip": "Output image size",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Number of edited images to generate",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "edit_image"
    CATEGORY = "api node/image/OpenAI Direct"
    DESCRIPTION = cleandoc(__doc__ or "")

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image."""
        # Handle batch dimension
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Convert from [0,1] to [0,255] and to uint8
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0

        tensor = tensor.clamp(0, 255).byte()

        # Convert to numpy array
        if tensor.shape[0] == 3:  # CHW format
            array = tensor.permute(1, 2, 0).cpu().numpy()
        else:  # HWC format
            array = tensor.cpu().numpy()

        return Image.fromarray(array)

    def pil_to_png_bytes(self, pil_image):
        """Convert PIL image to PNG bytes."""
        # Ensure image is square by resizing to the smaller dimension
        width, height = pil_image.size
        size = min(width, height)

        # Crop to square from center
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size

        pil_image = pil_image.crop((left, top, right, bottom))

        # Convert to RGBA if not already (for transparency support)
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

        # Save to bytes
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()

    def create_mask_from_alpha(self, image_tensor):
        """Create a mask from the alpha channel of an image tensor."""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]

        # Check if image has alpha channel
        if image_tensor.shape[0] == 4 or image_tensor.shape[2] == 4:
            if image_tensor.shape[0] == 4:  # CHW format
                alpha = image_tensor[3]
            else:  # HWC format
                alpha = image_tensor[:, :, 3]

            # Create mask: transparent areas (alpha < 0.5) become white (to be edited)
            mask = (alpha < 0.5).float()

            # Convert to 3-channel image (RGB)
            mask_rgb = mask.unsqueeze(-1).repeat(1, 1, 3)

            return mask_rgb.unsqueeze(0)  # Add batch dimension

        return None

    def edit_image(
        self,
        api_key: str,
        image,
        prompt: str,
        proxy_url: str = "",
        mask=None,
        size: str = "1024x1024",
        n: int = 1,
    ):
        if not api_key or not api_key.startswith("sk-"):
            raise ValueError("Please provide a valid OpenAI API key (starts with 'sk-')")

        if not prompt.strip():
            raise ValueError("Please provide a prompt describing the desired edit")

        try:
            # Setup proxy - either user-provided or auto-detected
            proxies = {}
            if proxy_url.strip():
                # User provided proxy
                proxies = {
                    'http': proxy_url.strip(),
                    'https': proxy_url.strip()
                }
                print(f"Using user-specified proxy: {proxy_url.strip()}")
            else:
                # Try to auto-detect system proxy
                proxies = get_system_proxy()
                if proxies:
                    print(f"Using auto-detected proxy: {proxies.get('http', 'Unknown')}")
                else:
                    print("No proxy configured, attempting direct connection")

            # Convert input image to PIL and then to PNG bytes
            pil_image = self.tensor_to_pil(image)
            image_bytes = self.pil_to_png_bytes(pil_image)

            # Prepare the multipart form data
            files = {
                'image': ('image.png', image_bytes, 'image/png'),
                'prompt': (None, prompt),
                'n': (None, str(n)),
                'size': (None, size),
            }

            # Handle mask if provided
            if mask is not None:
                pil_mask = self.tensor_to_pil(mask)
                mask_bytes = self.pil_to_png_bytes(pil_mask)
                files['mask'] = ('mask.png', mask_bytes, 'image/png')

            headers = {
                "Authorization": f"Bearer {api_key}",
            }

            # Make the API request
            response = requests.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                files=files,
                proxies=proxies,
                timeout=60
            )

            if response.status_code != 200:
                error_msg = f"OpenAI API error ({response.status_code})"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)

            result = response.json()

            if "data" not in result or not result["data"]:
                raise Exception("No edited images returned from OpenAI API")

            # Download and process edited images
            images = []
            print(f"API returned {len(result['data'])} edited image(s)")
            for i, img_data in enumerate(result["data"]):
                if "url" not in img_data:
                    print(f"Edited image {i}: No URL found in response data")
                    continue

                img_url = img_data["url"]
                print(f"Downloading edited image {i} from: {img_url[:50]}...")

                try:
                    img_response = requests.get(img_url, proxies=proxies, timeout=30)
                    print(f"Edited image {i} download response: {img_response.status_code}")

                    if img_response.status_code == 200:
                        # Convert to PIL Image
                        img = Image.open(io.BytesIO(img_response.content))

                        # Convert to RGB if necessary
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Convert to tensor
                        img_array = np.array(img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                        images.append(img_tensor)
                        print(f"Successfully processed edited image {i}: {img.size}")

                    else:
                        print(f"Failed to download edited image {i}: HTTP {img_response.status_code}")
                        print(f"Response content: {img_response.text[:200]}")

                except Exception as download_error:
                    print(f"Error downloading edited image {i}: {download_error}")

            print(f"Successfully downloaded {len(images)} out of {len(result['data'])} edited images")

            if not images:
                raise Exception("Failed to download any edited images from OpenAI")

            # Concatenate all images into a batch
            result_tensor = torch.cat(images, dim=0)
            return (result_tensor,)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error when calling OpenAI API: {str(e)}")
        except Exception as e:
            if "OpenAI API error" in str(e) or "Network error" in str(e):
                raise
            else:
                raise Exception(f"Error editing image: {str(e)}")


class OpenAITextNode(ComfyNodeABC):
    """
    Base class for OpenAI text generation nodes.
    """

    RETURN_TYPES = (IO.STRING,)
    FUNCTION = "api_call"
    CATEGORY = "api node/text/OpenAI"
    API_NODE = True


class OpenAIChatNode(OpenAITextNode):
    """
    Node to generate text responses from an OpenAI model.
    """

    def __init__(self) -> None:
        """Initialize the chat node with a new session ID and empty history."""
        self.current_session_id: str = str(uuid.uuid4())
        self.history: dict[str, list[HistoryEntry]] = {}
        self.previous_response_id: Optional[str] = None

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text inputs to the model, used to generate a response.",
                    },
                ),
                "persist_context": (
                    IO.BOOLEAN,
                    {
                        "default": True,
                        "tooltip": "Persist chat context between calls (multi-turn conversation)",
                    },
                ),
                "model": model_field_to_node_input(
                    IO.COMBO,
                    OpenAICreateResponse,
                    "model",
                    enum_type=SupportedOpenAIModel,
                ),
            },
            "optional": {
                "images": (
                    IO.IMAGE,
                    {
                        "default": None,
                        "tooltip": "Optional image(s) to use as context for the model. To include multiple images, you can use the Batch Images node.",
                    },
                ),
                "files": (
                    "OPENAI_INPUT_FILES",
                    {
                        "default": None,
                        "tooltip": "Optional file(s) to use as context for the model. Accepts inputs from the OpenAI Chat Input Files node.",
                    },
                ),
                "advanced_options": (
                    "OPENAI_CHAT_CONFIG",
                    {
                        "default": None,
                        "tooltip": "Optional configuration for the model. Accepts inputs from the OpenAI Chat Advanced Options node.",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "Generate text responses from an OpenAI model."

    def get_result_response(
        self,
        response_id: str,
        include: Optional[list[Includable]] = None,
        auth_kwargs: Optional[dict[str, str]] = None,
    ) -> OpenAIResponse:
        """
        Retrieve a model response with the given ID from the OpenAI API.

        Args:
            response_id (str): The ID of the response to retrieve.
            include (Optional[List[Includable]]): Additional fields to include
                in the response. See the `include` parameter for Response
                creation above for more information.

        """
        return PollingOperation(
            poll_endpoint=ApiEndpoint(
                path=f"{RESPONSES_ENDPOINT}/{response_id}",
                method=HttpMethod.GET,
                request_model=EmptyRequest,
                response_model=OpenAIResponse,
                query_params={"include": include},
            ),
            completed_statuses=["completed"],
            failed_statuses=["failed"],
            status_extractor=lambda response: response.status,
            auth_kwargs=auth_kwargs,
        ).execute()

    def get_message_content_from_response(
        self, response: OpenAIResponse
    ) -> list[OutputContent]:
        """Extract message content from the API response."""
        for output in response.output:
            if output.root.type == "message":
                return output.root.content
        raise TypeError("No output message found in response")

    def get_text_from_message_content(
        self, message_content: list[OutputContent]
    ) -> str:
        """Extract text content from message content."""
        for content_item in message_content:
            if content_item.root.type == "output_text":
                return str(content_item.root.text)
        return "No text output found in response"

    def get_history_text(self, session_id: str) -> str:
        """Convert the entire history for a given session to JSON string."""
        return json.dumps(self.history[session_id])

    def display_history_on_node(self, session_id: str, node_id: str) -> None:
        """Display formatted chat history on the node UI."""
        render_spec = {
            "node_id": node_id,
            "component": "ChatHistoryWidget",
            "props": {
                "history": self.get_history_text(session_id),
            },
        }
        PromptServer.instance.send_sync(
            "display_component",
            render_spec,
        )

    def add_to_history(
        self, session_id: str, prompt: str, output_text: str, response_id: str
    ) -> None:
        """Add a new entry to the chat history."""
        if session_id not in self.history:
            self.history[session_id] = []
        self.history[session_id].append(
            {
                "prompt": prompt,
                "response": output_text,
                "response_id": response_id,
                "timestamp": time.time(),
            }
        )

    def parse_output_text_from_response(self, response: OpenAIResponse) -> str:
        """Extract text output from the API response."""
        message_contents = self.get_message_content_from_response(response)
        return self.get_text_from_message_content(message_contents)

    def generate_new_session_id(self) -> str:
        """Generate a new unique session ID."""
        return str(uuid.uuid4())

    def get_session_id(self, persist_context: bool) -> str:
        """Get the current or generate a new session ID based on context persistence."""
        return (
            self.current_session_id
            if persist_context
            else self.generate_new_session_id()
        )

    def tensor_to_input_image_content(
        self, image: torch.Tensor, detail_level: Detail = "auto"
    ) -> InputImageContent:
        """Convert a tensor to an input image content object."""
        return InputImageContent(
            detail=detail_level,
            image_url=f"data:image/png;base64,{tensor_to_base64_string(image)}",
            type="input_image",
        )

    def create_input_message_contents(
        self,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        files: Optional[list[InputFileContent]] = None,
    ) -> InputMessageContentList:
        """Create a list of input message contents from prompt and optional image."""
        content_list: list[InputContent] = [
            InputTextContent(text=prompt, type="input_text"),
        ]
        if image is not None:
            for i in range(image.shape[0]):
                content_list.append(
                    self.tensor_to_input_image_content(image[i].unsqueeze(0))
                )
        if files is not None:
            content_list.extend(files)

        return InputMessageContentList(
            root=content_list,
        )

    def parse_response_id_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract response ID from prompt if it exists."""
        parsed_id = re.search(STARTING_POINT_ID_PATTERN, prompt)
        return parsed_id.group(1) if parsed_id else None

    def strip_response_tag_from_prompt(self, prompt: str) -> str:
        """Remove the response ID tag from the prompt."""
        return re.sub(STARTING_POINT_ID_PATTERN, "", prompt.strip())

    def delete_history_after_response_id(
        self, new_start_id: str, session_id: str
    ) -> None:
        """Delete history entries after a specific response ID."""
        if session_id not in self.history:
            return

        new_history = []
        i = 0
        while (
            i < len(self.history[session_id])
            and self.history[session_id][i]["response_id"] != new_start_id
        ):
            new_history.append(self.history[session_id][i])
            i += 1

        # Since it's the new starting point (not the response being edited), we include it as well
        if i < len(self.history[session_id]):
            new_history.append(self.history[session_id][i])

        self.history[session_id] = new_history

    def api_call(
        self,
        prompt: str,
        persist_context: bool,
        model: SupportedOpenAIModel,
        unique_id: Optional[str] = None,
        images: Optional[torch.Tensor] = None,
        files: Optional[list[InputFileContent]] = None,
        advanced_options: Optional[CreateModelResponseProperties] = None,
        **kwargs,
    ) -> tuple[str]:
        # Validate inputs
        validate_string(prompt, strip_whitespace=False)

        session_id = self.get_session_id(persist_context)
        response_id_override = self.parse_response_id_from_prompt(prompt)
        if response_id_override:
            is_starting_from_beginning = response_id_override == "start"
            if is_starting_from_beginning:
                self.history[session_id] = []
                previous_response_id = None
            else:
                previous_response_id = response_id_override
                self.delete_history_after_response_id(response_id_override, session_id)
            prompt = self.strip_response_tag_from_prompt(prompt)
        elif persist_context:
            previous_response_id = self.previous_response_id
        else:
            previous_response_id = None

        # Create response
        create_response = SynchronousOperation(
            endpoint=ApiEndpoint(
                path=RESPONSES_ENDPOINT,
                method=HttpMethod.POST,
                request_model=OpenAICreateResponse,
                response_model=OpenAIResponse,
            ),
            request=OpenAICreateResponse(
                input=[
                    Item(
                        root=InputMessage(
                            content=self.create_input_message_contents(
                                prompt, images, files
                            ),
                            role="user",
                        )
                    ),
                ],
                store=True,
                stream=False,
                model=model,
                previous_response_id=previous_response_id,
                **(
                    advanced_options.model_dump(exclude_none=True)
                    if advanced_options
                    else {}
                ),
            ),
            auth_kwargs=kwargs,
        ).execute()
        response_id = create_response.id

        # Get result output
        result_response = self.get_result_response(response_id, auth_kwargs=kwargs)
        output_text = self.parse_output_text_from_response(result_response)

        # Update history
        self.add_to_history(session_id, prompt, output_text, response_id)
        self.display_history_on_node(session_id, unique_id)
        self.previous_response_id = response_id

        return (output_text,)


class OpenAIInputFiles(ComfyNodeABC):
    """
    Loads and formats input files for OpenAI API.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """
        For details about the supported file input types, see:
        https://platform.openai.com/docs/guides/pdf-files?api-mode=responses
        """
        input_dir = folder_paths.get_input_directory()
        input_files = [
            f
            for f in os.scandir(input_dir)
            if f.is_file()
            and (f.name.endswith(".txt") or f.name.endswith(".pdf"))
            and f.stat().st_size < 32 * 1024 * 1024
        ]
        input_files = sorted(input_files, key=lambda x: x.name)
        input_files = [f.name for f in input_files]
        return {
            "required": {
                "file": (
                    IO.COMBO,
                    {
                        "tooltip": "Input files to include as context for the model. Only accepts text (.txt) and PDF (.pdf) files for now.",
                        "options": input_files,
                        "default": input_files[0] if input_files else None,
                    },
                ),
            },
            "optional": {
                "OPENAI_INPUT_FILES": (
                    "OPENAI_INPUT_FILES",
                    {
                        "tooltip": "An optional additional file(s) to batch together with the file loaded from this node. Allows chaining of input files so that a single message can include multiple input files.",
                        "default": None,
                    },
                ),
            },
        }

    DESCRIPTION = "Loads and prepares input files (text, pdf, etc.) to include as inputs for the OpenAI Chat Node. The files will be read by the OpenAI model when generating a response. 🛈 TIP: Can be chained together with other OpenAI Input File nodes."
    RETURN_TYPES = ("OPENAI_INPUT_FILES",)
    FUNCTION = "prepare_files"
    CATEGORY = "api node/text/OpenAI"

    def create_input_file_content(self, file_path: str) -> InputFileContent:
        return InputFileContent(
            file_data=text_filepath_to_data_uri(file_path),
            filename=os.path.basename(file_path),
            type="input_file",
        )

    def prepare_files(
        self, file: str, OPENAI_INPUT_FILES: list[InputFileContent] = []
    ) -> tuple[list[InputFileContent]]:
        """
        Loads and formats input files for OpenAI API.
        """
        file_path = folder_paths.get_annotated_filepath(file)
        input_file_content = self.create_input_file_content(file_path)
        files = [input_file_content] + OPENAI_INPUT_FILES
        return (files,)


class OpenAIChatConfig(ComfyNodeABC):
    """Allows setting additional configuration for the OpenAI Chat Node."""

    RETURN_TYPES = ("OPENAI_CHAT_CONFIG",)
    FUNCTION = "configure"
    DESCRIPTION = (
        "Allows specifying advanced configuration options for the OpenAI Chat Nodes."
    )
    CATEGORY = "api node/text/OpenAI"

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "truncation": (
                    IO.COMBO,
                    {
                        "options": ["auto", "disabled"],
                        "default": "auto",
                        "tooltip": "The truncation strategy to use for the model response. auto: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.disabled: If a model response will exceed the context window size for a model, the request will fail with a 400 error",
                    },
                ),
            },
            "optional": {
                "max_output_tokens": model_field_to_node_input(
                    IO.INT,
                    OpenAICreateResponse,
                    "max_output_tokens",
                    min=16,
                    default=4096,
                    max=16384,
                    tooltip="An upper bound for the number of tokens that can be generated for a response, including visible output tokens",
                ),
                "instructions": model_field_to_node_input(
                    IO.STRING, OpenAICreateResponse, "instructions", multiline=True
                ),
            },
        }

    def configure(
        self,
        truncation: bool,
        instructions: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
    ) -> tuple[CreateModelResponseProperties]:
        """
        Configure advanced options for the OpenAI Chat Node.

        Note:
            While `top_p` and `temperature` are listed as properties in the
            spec, they are not supported for all models (e.g., o4-mini).
            They are not exposed as inputs at all to avoid having to manually
            remove depending on model choice.
        """
        return (
            CreateModelResponseProperties(
                instructions=instructions,
                truncation=truncation,
                max_output_tokens=max_output_tokens,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "OpenAIDalle2": OpenAIDalle2,
    "OpenAIDalle3": OpenAIDalle3,
    "OpenAIGPTImage1": OpenAIGPTImage1,
    "OpenAIGPTImage1Direct": OpenAIGPTImage1Direct,
    "OpenAIGPTImage1Direct_Simple": OpenAIGPTImage1Direct_Simple,
    "OpenAIImageEditDirect": OpenAIImageEditDirect,
    "OpenAIChatNode": OpenAIChatNode,
    "OpenAIInputFiles": OpenAIInputFiles,
    "OpenAIChatConfig": OpenAIChatConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIDalle2": "OpenAI DALL·E 2",
    "OpenAIDalle3": "OpenAI DALL·E 3",
    "OpenAIGPTImage1": "OpenAI GPT Image 1",
    "OpenAIGPTImage1Direct": "OpenAI GPT Image 1 (Direct API)",
    "OpenAIGPTImage1Direct_Simple": "OpenAI GPT Image 1 Direct (Simple)",
    "OpenAIImageEditDirect": "OpenAI Image Editor (Direct API)",
    "OpenAIChatNode": "OpenAI Chat",
    "OpenAIInputFiles": "OpenAI Chat Input Files",
    "OpenAIChatConfig": "OpenAI Chat Advanced Options",
}
