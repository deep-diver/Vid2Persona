from typing import Union, Iterable

import vertexai
from vertexai.generative_models import (
    GenerativeModel, Part, 
    GenerationResponse, GenerationConfig
)

from .utils import parse_first_json_snippet

def _default_gen_config():
    return GenerationConfig(
        max_output_tokens=2048,
        temperature=0.4,
        top_p=1,
        top_k=32
    )

def init_vertexai(project_id: str, location: str) -> None:
    vertexai.init(project=project_id, location=location)

async def _ask_about_video(
    prompt: str="What is in the video?",
    gen_config: dict=_default_gen_config(),
    model_name: str="gemini-1.0-pro-vision",
    gcs: str=None, 
    base64_content: bytes=None
) -> Union[GenerationResponse, Iterable[GenerationResponse]]:
    if gcs is None and base64_content is None:
        raise ValueError("Either a GCS bucket path or base64_encoded string of the video must be provided")

    if gcs is not None and base64_content is not None:
        raise ValueError("Only one of gcs or base64_encoded must be provided")

    if gcs is not None:
        video = Part.from_uri(gcs, mime_type="video/mp4")
    else:
        video = Part.from_data(data=base64_content, mime_type="video/mp4")

    model = GenerativeModel(model_name)
    return await model.generate_content_async(
        [video, prompt],
        generation_config=gen_config
    )

async def ask_about_video(prompt: str, video_clip: bytes, retry_num: int=10):
    json_content = None
    cur_retry = 0

    while json_content is None and cur_retry < retry_num:
        try:
            resps = await _ask_about_video(
                prompt=prompt, base64_content=video_clip
            )

            json_content = parse_first_json_snippet(resps.text)
        except Exception as e:
            cur_retry = cur_retry + 1
            print(f"......retry {e}")

    return json_content