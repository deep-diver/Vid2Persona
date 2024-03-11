import asyncio
import argparse
from vid2persona import init
from vid2persona.pipeline import vlm
from vid2persona.pipeline import llm

def validate_args(func):
    def inner_function(args):
        gcp_project_id, gcp_project_location = init.get_env_vars()

        if args.gcp_project_id is None: args.gcp_project_id = gcp_project_id
        if args.gcp_project_location is None: args.gcp_project_location = gcp_project_location

        if args.gcp_project_id is None or args.gcp_project_location is None:
            raise ValueError("gcp-project-id or gcp-project-location is missing")
        
        if args.hf_access_token is not None:            
            if args.model_id not in init.ALLOWED_LLM_FOR_HF_PRO_ACCOUNTS:
                raise ValueError("not supported model for Hugging Face PRO account")
        return func(args)
    return inner_function

@validate_args
async def workflow(args):
    traits = await vlm.get_traits(
        args.gcp_project_id, 
        args.gcp_project_location, 
        args.target_movie_clip, 
        args.prompt_tpl_path
    )
    if 'characters' in traits:
        traits = traits['characters'][0]

    messages = []
    async for response in llm.chat(
        args.message, messages, traits,
        args.prompt_tpl_path, args.model_id, 
        args.max_input_token_length, args.max_new_tokens,
        args.temperature, args.top_p, args.top_k, 
        args.repetition_penalty, hf_token=args.hf_access_token
    ):
        print(response, end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcp-project-id', type=str, required=True)
    parser.add_argument('--gcp-project-location', type=str, required=True)
    parser.add_argument('--target-movie-clip', type=str, default="assets/sample1.mp4")
    parser.add_argument('--prompt-tpl-path', type=str, default="vid2persona/prompts")
    parser.add_argument('--hf-access-token', type=str, default=None)
    parser.add_argument('--model-id', type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument('--max-input-token-length', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--repetition-penalty', type=float, default=1.2)
    parser.add_argument('--message', type=str, default="Hello there! How are you doing?")
    args = parser.parse_args()

    asyncio.run(workflow(args))
