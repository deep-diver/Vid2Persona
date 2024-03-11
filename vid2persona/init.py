import os

# https://huggingface.co/blog/inference-pro
ALLOWED_LLM_FOR_HF_PRO_ACCOUNTS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "openchat/openchat-3.5-0106"
]

def get_env_vars():
    gcp_project_id = os.getenv("GCP_PROJECT_ID", None)
    gcp_project_loc = os.getenv("GCP_PROJECT_LOCATION", None)

    return gcp_project_id, gcp_project_loc