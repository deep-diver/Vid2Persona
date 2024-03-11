from openai import AsyncOpenAI

async def send_messages(
    messages: list, 
    model_id: str, 
    hf_token: str, 
    parameters: dict
):
    parameters.pop('top_k')
    parameters['max_tokens'] = parameters.pop('max_new_token')
    parameters['presence_penalty'] = parameters.pop('repetition_penalty')

    client = AsyncOpenAI(
        base_url=f"https://api-inference.huggingface.co/models/{model_id}/v1",
        api_key=hf_token,
    )

    responses = await client.chat.completions.create(
        model=model_id, messages=messages, stream=True
    )

    async for response in responses:
        yield response.choices[0].delta.content