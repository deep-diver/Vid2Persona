import toml
from vid2persona.gen.gemini import init_vertexai, ask_about_video
from vid2persona.utils import get_base64_content

def get_traits(
    gcp_project_id: str, gcp_project_location: str,
    video_clip_path: str, prompt_tpl_path: str,
):
    prompt_tpl_path = f"{prompt_tpl_path}/vlm.toml"
    prompt = toml.load(prompt_tpl_path)['extraction']['traits']
    init_vertexai(gcp_project_id, gcp_project_location)
    video_clip = get_base64_content(video_clip_path)

    response = ask_about_video(prompt=prompt, video_clip=video_clip)
    return response
