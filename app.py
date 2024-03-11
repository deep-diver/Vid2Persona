import argparse
import gradio as gr

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

async def extract_traits(video_path,  gcp_project_id, gcp_project_location, prompt_tpl_path):
    traits = await vlm.get_traits(
        gcp_project_id, 
        gcp_project_location, 
        video_path,
        prompt_tpl_path
    )
    if 'characters' in traits:
        traits = traits['characters'][0]

    return [
        traits, [], 
        gr.Textbox("", interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
        gr.Button(interactive=True)
    ]

async def conversation(
    message: str, messages: list, traits: dict,
    prompt_tpl_path: str, model_id: str, 
    max_input_token_length: int, max_new_tokens: int,
    temperature: float, top_p: float, top_k: float,
    repetition_penalty: float, hf_access_token: str, 
):
    if hf_access_token == "": 
        hf_access_token = None

    messages = messages + [[message, ""]]
    yield [messages, message, gr.Button(interactive=False), gr.Button(interactive=False)]

    async for partial_response in llm.chat(
        message, messages, traits,
        prompt_tpl_path, model_id, 
        max_input_token_length, max_new_tokens,
        temperature, top_p, top_k, 
        repetition_penalty, hf_token=hf_access_token
    ):
        last_message = messages[-1]
        last_message[1] = last_message[1] + partial_response
        messages[-1] = last_message
        yield [messages, "", gr.Button(interactive=False), gr.Button(interactive=False)]

    yield [messages, "", gr.Button(interactive=True), gr.Button(interactive=True)]

async def regen_conversation(
    messages: list, traits: dict,
    prompt_tpl_path: str, model_id: str, 
    max_input_token_length: int, max_new_tokens: int,
    temperature: float, top_p: float, top_k: float,
    repetition_penalty: float, hf_access_token: str, 
):
    if len(messages) > 0:
        message = messages[-1][0]
        messages = messages[:-1]
        messages = messages + [[message, ""]]
        yield [messages, "", gr.Button(interactive=False), gr.Button(interactive=False)]

        async for partial_response in llm.chat(
            message, messages, traits,
            prompt_tpl_path, model_id, 
            max_input_token_length, max_new_tokens,
            temperature, top_p, top_k, 
            repetition_penalty, hf_token=hf_access_token
        ):
            last_message = messages[-1]
            last_message[1] = last_message[1] + partial_response
            messages[-1] = last_message
            yield [messages, "", gr.Button(interactive=False), gr.Button(interactive=False)]

        yield [messages, "", gr.Button(interactive=True), gr.Button(interactive=True)]

@validate_args
def main(args):
    with gr.Blocks(css="styles.css", theme=gr.themes.Soft()) as demo:
        # hidden components
        gcp_project_id = gr.Textbox(args.gcp_project_id, visible=False)
        gcp_project_location = gr.Textbox(args.gcp_project_location, visible=False)
        prompt_tpl_path = gr.Textbox(args.prompt_tpl_path, visible=False)
        hf_access_token = gr.Textbox(args.hf_access_token, visible=False)

        gr.Markdown("Vid2Persona", elem_classes=["md-center", "h1-font"])
        gr.Markdown("This project breathes life into video characters by using AI to describe their personality and then chat with you as them.")

        with gr.Column(elem_classes=["group"]):
            with gr.Row():
                video = gr.Video(label="upload short video clip")
                traits = gr.Json(label="extracted traits")
            
            with gr.Row():
                trait_gen = gr.Button("generate  traits")

        with gr.Column(elem_classes=["group"]):
            chatbot = gr.Chatbot([], label="chatbot", elem_id="chatbot", elem_classes=["chatbot-no-label"])
            with gr.Row():
                clear = gr.Button("clear conversation", interactive=False)
                regen = gr.Button("regenerate the last", interactive=False)
                stop = gr.Button("stop", interactive=False) 
            user_input = gr.Textbox(placeholder="ask anything", interactive=False, elem_classes=["textbox-no-label", "textbox-no-top-bottom-borders"])

            with gr.Accordion("parameters' control pane", open=False):
                model_id = gr.Dropdown(choices=init.ALLOWED_LLM_FOR_HF_PRO_ACCOUNTS, value=args.model_id, label="Model ID")

                with gr.Row():
                    max_input_token_length = gr.Slider(minimum=1024, maximum=4096, value=args.max_input_token_length, label="max-input-tokens")
                    max_new_tokens = gr.Slider(minimum=512, maximum=2048, value=args.max_new_tokens, label="max-new-tokens")

                with gr.Row():
                    temperature = gr.Slider(minimum=0, maximum=2, step=0.1, value=args.temperature, label="temperature")
                    top_p = gr.Slider(minimum=0, maximum=2, step=0.1, value=args.top_p, label="top-p")
                    top_k = gr.Slider(minimum=0, maximum=2, step=0.1, value=args.top_k, label="top-k")
                    repetition_penalty = gr.Slider(minimum=0, maximum=2, step=0.1, value=args.repetition_penalty, label="repetition-penalty")
        
        with gr.Row():
            gr.Markdown(
                "[![GitHub Repo](https://img.shields.io/badge/GitHub%20Repo-gray?style=for-the-badge&logo=github&link=https://github.com/deep-diver/Vid2Persona)](https://github.com/deep-diver/Vid2Persona) "
                "[![Chansung](https://img.shields.io/badge/Chansung-blue?style=for-the-badge&logo=twitter&link=https://twitter.com/algo_diver)](https://twitter.com/algo_diver) "
                "[![Sayak](https://img.shields.io/badge/Sayak-blue?style=for-the-badge&logo=twitter&link=https://twitter.com/RisingSayak)](https://twitter.com/RisingSayak )",
                elem_id="bottom-md"
            )

        trait_gen.click(
            extract_traits,
            [video, gcp_project_id, gcp_project_location, prompt_tpl_path],
            [traits, chatbot, user_input, clear, regen, stop]
        )

        conv = user_input.submit(
            conversation,
            [
                user_input, chatbot, traits,
                prompt_tpl_path, model_id, 
                max_input_token_length, max_new_tokens,
                temperature, top_p, top_k,
                repetition_penalty, hf_access_token
            ],
            [chatbot, user_input, clear, regen]
        )

        clear.click(
            lambda: [
                gr.Chatbot([]),
                gr.Button(interactive=False),
                gr.Button(interactive=False),
            ],
            None, [chatbot, clear, regen]
        )

        conv_regen = regen.click(
            regen_conversation,
            [
                chatbot, traits,
                prompt_tpl_path, model_id, 
                max_input_token_length, max_new_tokens,
                temperature, top_p, top_k,
                repetition_penalty, hf_access_token
            ],
            [chatbot, user_input, clear, regen]
        )

        stop.click(
            None, None, None, 
            cancels=[conv, conv_regen]
        )

    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcp-project-id', type=str, required=True, 
                        help="The ID of your Google Cloud Platform (GCP) project. which "
                             "you want to run Vertex AI multimodal video anlaysis with "
                             "Gemini 1.0 Pro Vision model.")
    
    parser.add_argument('--gcp-project-location', type=str, required=True, 
                        help="The GCP region where you want to run Vertex AI multimodal "
                             "video analysis with Gemini 1.0 Pro Vision model.")
        
    parser.add_argument('--prompt-tpl-path', type=str, default="vid2persona/prompts", 
                        help="Path to the directory containing prompt templates for the model.")
    
    parser.add_argument('--hf-access-token', type=str, default=None, 
                        help="Your Hugging Face access token (if needed for model access). "
                             "If you don't specify this, the program will run the model on "
                             "your local machine")
    
    parser.add_argument('--model-id', type=str, default="HuggingFaceH4/zephyr-7b-beta", 
                        help="The Hugging Face model repository fo the language model to use.")
    
    parser.add_argument('--max-input-token-length', type=int, default=4096, 
                        help="Maximum number of input tokens allowed for the model.")
    
    parser.add_argument('--max-new-tokens', type=int, default=2048, 
                        help="Maximum number of input tokens allowed for the model.")
    
    parser.add_argument('--temperature', type=float, default=0.6, 
                        help="Controls the randomness/creativity of the model's output "
                             "(higher values mean more random).")
    
    parser.add_argument('--top-p', type=float, default=0.9, 
                        help="Nucleus sampling: considers the smallest set of tokens with "
                             "a cumulative probability of at least 'top_p'.")
    
    parser.add_argument('--top-k', type=int, default=50, 
                        help="Limits the number of tokens considered for generation at each step.")
    
    parser.add_argument('--repetition-penalty', type=float, default=1.2, 
                        help="Penalizes repeated tokens to encourage diversity in the output.")
    
    args = parser.parse_args()
    main(args)
