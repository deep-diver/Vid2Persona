import toml
from string import Template
from transformers import AutoTokenizer

from vid2persona.gen import tgi_openllm
from vid2persona.gen import local_openllm

tokenizer = None

def _get_system_prompt(
    personality_json_dict: dict,
    prompt_tpl_path: str
) -> str:
    """Assumes a single character is passed."""
    prompt_tpl_path = f"{prompt_tpl_path}/llm.toml"
    system_prompt = Template(toml.load(prompt_tpl_path)['conversation']['system'])

    name = personality_json_dict["name"]
    physcial_description = personality_json_dict["physicalDescription"]
    personality_traits = [str(trait) for trait in personality_json_dict["personalityTraits"]]
    likes = [str(like) for like in personality_json_dict["likes"]]
    dislikes = [str(dislike) for dislike in personality_json_dict["dislikes"]]
    background = [str(info) for info in personality_json_dict["background"]]
    goals = [str(goal) for goal in personality_json_dict["goals"]]
    relationships = [str(relationship) for relationship in personality_json_dict["relationships"]]

    system_prompt = system_prompt.substitute(
        name=name,
        physcial_description=physcial_description,
        personality_traits=', '.join(personality_traits),
        likes=', '.join(likes),
        background=', '.join(background),
        goals=', '.join(goals),
        relationships=', '.join(relationships)
    )

    return system_prompt

async def chat(
    message: str,
    chat_history: list[tuple[str, str]],
    personality_json_dict: dict,
    prompt_tpl_path: str,

    model_id: str,
    max_input_token_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,

    hf_token: str,
):
    messages = []
    system_prompt = _get_system_prompt(personality_json_dict, prompt_tpl_path)
    messages.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    messages.append({"role": "user", "content": message})

    parameters = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty
    }

    if hf_token is None:
        for response in local_openllm.send_message(messages, model_id, max_input_token_length, parameters):
            yield response
    else:
        async for response in tgi_openllm.send_messages(messages, model_id, hf_token, parameters):
            yield response