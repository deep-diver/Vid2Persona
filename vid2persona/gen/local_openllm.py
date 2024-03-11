import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer

model = None
tokenizer = None

def send_message(
    messages: list,
    model_id: str,
    max_input_token_length: int,
    parameters: dict
):
    global tokenizer
    global model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.use_default_system_prompt = False
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")
    if input_ids.shape[1] > max_input_token_length:
        input_ids = input_ids[:, -max_input_token_length:]
        print(f"Trimmed input from conversation as it was longer than {max_input_token_length} tokens.")
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        do_sample=True,
        num_beams=1,
        **parameters
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    for text in streamer:
        yield text.replace("<|assistant|>", "")