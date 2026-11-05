# OpenAI API Models for ComfyUI Custom Nodes
# Simplified version containing only OpenAI-related classes

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, RootModel, StrictBytes


class Background(str, Enum):
    opaque = 'opaque'
    transparent = 'transparent'


class Moderation(str, Enum):
    auto = 'auto'
    disabled = 'disabled'


class Detail(str, Enum):
    low = 'low'
    high = 'high'
    auto = 'auto'


class Type10(str, Enum):
    input_text = 'input_text'


class Type11(str, Enum):
    input_image = 'input_image'


class Type12(str, Enum):
    input_file = 'input_file'


class Type13(str, Enum):
    output_audio = 'output_audio'


class Type14(str, Enum):
    message = 'message'


class Type15(str, Enum):
    output_text = 'output_text'


class Type9(str, Enum):
    message = 'message'


class Role(str, Enum):
    user = 'user'
    assistant = 'assistant'
    system = 'system'


class Role3(str, Enum):
    user = 'user'
    assistant = 'assistant'
    system = 'system'


class Role4(str, Enum):
    assistant = 'assistant'


class Status2(str, Enum):
    incomplete = 'incomplete'


class Object(str, Enum):
    response = 'response'


class Reason(str, Enum):
    max_output_tokens = 'max_output_tokens'
    max_completion_tokens = 'max_completion_tokens'


class Includable(str, Enum):
    file_search_call_results = 'file_search_call.results'
    message_input_image_image_url = 'message.input_image.image_url'
    computer_call_output_output_image_url = 'computer_call_output.output.image_url'


class OpenAIModels(str, Enum):
    gpt_4 = 'gpt-4'
    gpt_4_0314 = 'gpt-4-0314'
    gpt_4_0613 = 'gpt-4-0613'
    gpt_4_32k = 'gpt-4-32k'
    gpt_4_32k_0314 = 'gpt-4-32k-0314'
    gpt_4_32k_0613 = 'gpt-4-32k-0613'
    gpt_4_0125_preview = 'gpt-4-0125-preview'
    gpt_4_turbo = 'gpt-4-turbo'
    gpt_4_turbo_2024_04_09 = 'gpt-4-turbo-2024-04-09'
    gpt_4_turbo_preview = 'gpt-4-turbo-preview'
    gpt_4_1106_preview = 'gpt-4-1106-preview'
    gpt_4_vision_preview = 'gpt-4-vision-preview'
    gpt_4o = 'gpt-4o'
    gpt_4o_2024_05_13 = 'gpt-4o-2024-05-13'
    gpt_4o_2024_08_06 = 'gpt-4o-2024-08-06'
    gpt_4o_2024_11_20 = 'gpt-4o-2024-11-20'
    gpt_4o_mini = 'gpt-4o-mini'
    gpt_4o_mini_2024_07_18 = 'gpt-4o-mini-2024-07-18'
    o1_preview = 'o1-preview'
    o1_preview_2024_09_12 = 'o1-preview-2024-09-12'
    o1_mini = 'o1-mini'
    o1_mini_2024_09_12 = 'o1-mini-2024-09-12'
    o1 = 'o1'
    o1_2024_12_17 = 'o1-2024-12-17'
    o1_pro = 'o1-pro'
    o1_pro_2024_12_17 = 'o1-pro-2024-12-17'
    o3_mini = 'o3-mini'
    o3_mini_2025_01_31 = 'o3-mini-2025-01-31'
    o4_mini = 'o4-mini'
    o4_mini_2025_01_31 = 'o4-mini-2025-01-31'
    gpt_3_5_turbo = 'gpt-3.5-turbo'
    gpt_3_5_turbo_16k = 'gpt-3.5-turbo-16k'
    gpt_3_5_turbo_0301 = 'gpt-3.5-turbo-0301'
    gpt_3_5_turbo_0613 = 'gpt-3.5-turbo-0613'
    gpt_3_5_turbo_1106 = 'gpt-3.5-turbo-1106'
    gpt_3_5_turbo_0125 = 'gpt-3.5-turbo-0125'
    gpt_3_5_turbo_16k_0613 = 'gpt-3.5-turbo-16k-0613'
    gpt_3_5_turbo_instruct = 'gpt-3.5-turbo-instruct'
    gpt_3_5_turbo_instruct_0914 = 'gpt-3.5-turbo-instruct-0914'
    text_davinci_003 = 'text-davinci-003'
    text_davinci_002 = 'text-davinci-002'
    code_davinci_002 = 'code-davinci-002'


class InputTextContent(BaseModel):
    text: str = Field(..., description='The text input to the model.')
    type: Type10 = Field(
        ..., description='The type of the input item. Always `input_text`.'
    )


class InputImageContent(BaseModel):
    detail: Detail = Field(
        ...,
        description='The detail level of the image to be sent to the model. One of `high`, `low`, or `auto`. Defaults to `auto`.',
    )
    file_id: Optional[str] = Field(
        None, description='The ID of the file to be sent to the model.'
    )
    image_url: Optional[str] = Field(
        None, description='The URL of the image to be sent to the model.'
    )
    type: Type11 = Field(
        ..., description='The type of the input item. Always `input_image`.'
    )


class InputFileContent(BaseModel):
    file_data: Optional[str] = Field(
        None, description='The content of the file to be sent to the model.\n'
    )
    file_id: Optional[str] = Field(
        None, description='The ID of the file to be sent to the model.'
    )
    filename: Optional[str] = Field(
        None, description='The name of the file to be sent to the model.'
    )
    type: Type12 = Field(
        ..., description='The type of the input item. Always `input_file`.'
    )


class OutputTextContent(BaseModel):
    text: str = Field(..., description='The text content')
    type: Type15 = Field(..., description='The type of output content')


class OutputAudioContent(BaseModel):
    data: str = Field(..., description='Base64-encoded audio data')
    transcript: str = Field(..., description='Transcript of the audio')
    type: Type13 = Field(..., description='The type of output content')


class InputContent(
    RootModel[Union[InputTextContent, InputImageContent, InputFileContent]]
):
    root: Union[InputTextContent, InputImageContent, InputFileContent]


class InputMessageContentList(RootModel[List[InputContent]]):
    root: List[InputContent] = Field(
        ...,
        description='A list of one or many input items to the model, containing different content \ntypes.\n',
        title='Input item content list',
    )


class OutputContent(RootModel[Union[OutputTextContent, OutputAudioContent]]):
    root: Union[OutputTextContent, OutputAudioContent]


class InputMessage(BaseModel):
    content: Optional[InputMessageContentList] = None
    role: Optional[Role3] = None
    status: Optional[Status2] = None
    type: Optional[Type9] = None


class OutputMessage(BaseModel):
    content: List[OutputContent] = Field(..., description='The content of the message')
    role: Role4 = Field(..., description='The role of the message')
    type: Type14 = Field(..., description='The type of output item')


class OpenAIImageEditRequest(BaseModel):
    background: Optional[str] = Field(
        None, description='Background transparency', examples=['opaque']
    )
    model: str = Field(
        ..., description='The model to use for image editing', examples=['gpt-image-1']
    )
    moderation: Optional[Moderation] = Field(
        None, description='Content moderation setting', examples=['auto']
    )
    n: Optional[int] = Field(
        None, description='Number of images to generate', examples=[1], ge=1, le=8
    )
    prompt: str = Field(
        ..., description='Text prompt for image editing', examples=['A cat wearing a hat']
    )
    quality: Optional[str] = Field(
        None, description='Image quality setting', examples=['low']
    )
    seed: Optional[int] = Field(
        None, description='Random seed for reproducibility', examples=[42]
    )
    size: Optional[str] = Field(
        None, description='Image size', examples=['1024x1024']
    )


class OpenAIImageGenerationRequest(BaseModel):
    background: Optional[Background] = Field(
        None, description='Background transparency', examples=['opaque']
    )
    model: Optional[str] = Field(
        None, description='The model to use for image generation', examples=['dall-e-3']
    )
    moderation: Optional[Moderation] = Field(
        None, description='Content moderation setting', examples=['auto']
    )
    n: Optional[int] = Field(
        None, description='Number of images to generate', examples=[1], ge=1, le=8
    )
    prompt: str = Field(
        ..., description='Text prompt for image generation', examples=['A cat wearing a hat']
    )
    quality: Optional[str] = Field(
        None, description='Image quality setting', examples=['standard']
    )
    seed: Optional[int] = Field(
        None, description='Random seed for reproducibility', examples=[42]
    )
    size: Optional[str] = Field(
        None, description='Image size', examples=['1024x1024']
    )
    style: Optional[str] = Field(
        None, description='Image style', examples=['natural']
    )


class Datum2(BaseModel):
    b64_json: Optional[str] = Field(None, description='Base64 encoded image data')
    revised_prompt: Optional[str] = Field(None, description='Revised prompt used for generation')
    url: Optional[str] = Field(None, description='URL of the generated image')


class InputTokensDetails(BaseModel):
    image_tokens: Optional[int] = None
    text_tokens: Optional[int] = None


class Usage(BaseModel):
    input_tokens_details: Optional[InputTokensDetails] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class OpenAIImageGenerationResponse(BaseModel):
    data: Optional[List[Datum2]] = None
    usage: Optional[Usage] = None


class IncompleteDetails(BaseModel):
    reason: Optional[Reason] = Field(
        None, description='The reason why the response is incomplete.'
    )


class ResponseError(BaseModel):
    code: Optional[str] = Field(None, description='Error code')
    message: Optional[str] = Field(None, description='Error message')


class InputTokensDetails1(BaseModel):
    cached_tokens: int = Field(
        ...,
        description='The number of tokens that were retrieved from the cache. \n[More on prompt caching](/docs/guides/prompt-caching).\n',
    )


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = Field(..., description='The number of reasoning tokens.')


class ResponseUsage(BaseModel):
    input_tokens: int = Field(..., description='The number of input tokens.')
    input_tokens_details: Optional[InputTokensDetails1] = None
    output_tokens: int = Field(..., description='The number of output tokens.')
    output_tokens_details: Optional[OutputTokensDetails] = None
    total_tokens: int = Field(..., description='The total number of tokens.')


class Status8(str, Enum):
    in_progress = 'in_progress'
    completed = 'completed'


class Type16(str, Enum):
    summary = 'summary'


class SummaryItem(BaseModel):
    text: str = Field(
        ...,
        description='A short summary of the reasoning used by the model when generating\nthe response.\n',
    )
    type: Type16 = Field(
        ..., description='The type of reasoning content. Always `summary`.\n'
    )


class ReasoningItem(BaseModel):
    id: str = Field(
        ..., description='The unique identifier of the reasoning content.\n'
    )
    status: Optional[Status8] = Field(
        None,
        description='The status of the reasoning content. Can be `in_progress` or `completed`.\n',
    )
    summary: Optional[SummaryItem] = None


class Reasoning(BaseModel):
    content: List[ReasoningItem] = Field(
        ..., description='The reasoning content generated by the model.\n'
    )


class Text(BaseModel):
    value: str = Field(..., description='The text value.\n')


class ToolChoiceOptions(str, Enum):
    auto = 'auto'
    none = 'none'
    required = 'required'


class ToolChoiceTypes(str, Enum):
    file_search = 'file_search'
    function = 'function'


class ToolChoiceFunction(BaseModel):
    function: Dict[str, Any] = Field(..., description='Function definition')
    type: str = Field(..., description='Tool type')


class CreateModelResponseProperties(BaseModel):
    instructions: Optional[str] = Field(
        None,
        description='Instructions for the model on how it should behave or respond.\n',
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description='An upper bound for the number of tokens that can be generated for a response, including visible output tokens and [reasoning tokens](/docs/guides/reasoning).\n',
    )
    model: Optional[OpenAIModels] = None
    previous_response_id: Optional[str] = Field(
        None,
        description='The unique ID of the previous response to the model. Use this to\ncreate multi-turn conversations. Learn more about \n[conversation state](/docs/guides/conversation-state).\n',
    )
    reasoning: Optional[Reasoning] = None
    text: Optional[Text] = None
    tool_choice: Optional[
        Union[ToolChoiceOptions, ToolChoiceTypes, ToolChoiceFunction]
    ] = Field(
        None,
        description='Controls which (if any) tool is called by the model.\n`none` means the model will not call any tool and instead generates a message.\n`auto` means the model can pick between generating a message or calling one or more tools.\n`required` means the model must call one or more tools.\nSpecifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.\n\n`none` is the default when no tools are present. `auto` is the default if tools are present.\n',
    )
    truncation: Optional[str] = Field(
        None,
        description='The truncation strategy to use for the model response.\n\n`auto`: If the context of this response and previous ones exceeds the\nmodel\'s context window size, the model will truncate the response to fit\nthe context window by dropping input items in the middle of the\nconversation.\n\n`disabled`: If a model response will exceed the context window size for a\nmodel, the request will fail with a 400 error.\n',
    )


class ResponseProperties(BaseModel):
    store: Optional[bool] = Field(
        None,
        description='Whether to store the response. Responses are stored for 30 days and can be used for [conversation state](/docs/guides/conversation-state).\n',
    )
    stream: Optional[bool] = Field(
        None, description='Whether to stream the response.\n'
    )


class ModelResponseProperties(BaseModel):
    output: Optional[List[Any]] = Field(
        None, description='The output generated by the model.\n'
    )
    status: Optional[str] = Field(None, description='The status of the response.\n')
    usage: Optional[ResponseUsage] = None


class EasyInputMessage(BaseModel):
    content: Union[str, InputMessageContentList] = Field(
        ...,
        description='Text, image, or audio input to the model, used to generate a response.\nCan also contain previous assistant responses.\n',
    )
    role: Role = Field(
        ..., description='The role of the message author. One of `user`, `assistant`, or `system`.\n'
    )


class FileSearchToolCall(BaseModel):
    # Placeholder for file search tool call
    pass


class FunctionToolCall(BaseModel):
    # Placeholder for function tool call
    pass


class Item(
    RootModel[
        Union[
            InputMessage,
            OutputMessage,
            FileSearchToolCall,
            FunctionToolCall,
        ]
    ]
):
    root: Union[
        InputMessage,
        OutputMessage,
        FileSearchToolCall,
        FunctionToolCall,
    ]


class OutputItem(
    RootModel[
        Union[
            OutputMessage,
            FileSearchToolCall,
            FunctionToolCall,
        ]
    ]
):
    root: Union[
        OutputMessage,
        FileSearchToolCall,
        FunctionToolCall,
    ]


class InputItem(RootModel[Union[EasyInputMessage, Item]]):
    root: Union[EasyInputMessage, Item]


class OpenAICreateResponse(CreateModelResponseProperties, ResponseProperties):
    include: Optional[List[Includable]] = Field(
        None,
        description='Specify additional output data to include in the model response. Currently\nsupported values are:\n- `file_search_call.results`: Include the search results of\n  the file search tool call.\n- `message.input_image.image_url`: Include image urls from the input message.\n- `computer_call_output.output.image_url`: Include image urls from the computer call output.\n',
    )
    input: Union[str, List[InputItem]] = Field(
        ...,
        description='Text, image, or file inputs to the model, used to generate a response.\n\nLearn more:\n- [Text inputs and outputs](/docs/guides/text)\n- [Image inputs](/docs/guides/images)\n- [File inputs](/docs/guides/pdf-files)\n- [Conversation state](/docs/guides/conversation-state)\n- [Function calling](/docs/guides/function-calling)\n',
    )
    parallel_tool_calls: Optional[bool] = Field(
        True, description='Whether to allow the model to run tool calls in parallel.\n'
    )
    tools: Optional[List[Any]] = Field(
        None,
        description='A list of tools the model may call. Currently, only `function` and `file_search` tools are supported.\n',
    )


class OpenAIResponse(ModelResponseProperties, ResponseProperties):
    created_at: Optional[float] = Field(
        None,
        description='Unix timestamp (in seconds) of when this Response was created.',
    )
    error: Optional[ResponseError] = None
    id: Optional[str] = Field(None, description='Unique identifier for this Response.')
    incomplete_details: Optional[IncompleteDetails] = Field(
        None, description='Details about why the response is incomplete.\n'
    )
    object: Optional[Object] = Field(
        None, description='The object type, which is always `response`.\n'
    )
