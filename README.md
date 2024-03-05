# Vid2Persona

This project breathes life into video characters by using AI to describe their personality and then chat with you as them.

## Brainstormed workflow

1. get a person's description from the video clip using Large Multimodal Model
    - We choose [Get video descriptions](https://cloud.google.com/vertex-ai/generative-ai/docs/video/video-descriptions#vid-desc-rest) service from [Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai).
      
2. based on the description, ask Large Language Model to pretend to be the person
3. then, chatting with that personality
    - We choose either [Gemini API from Google AI Studio](https://ai.google.dev/) or [Gemini API from Generative AI on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini).

Optionally, we could leverage other open source technologies
- [diffusers](https://huggingface.co/docs/diffusers/en/index) to generate images of the person in different poses or the backgrounds
- [transformers](https://huggingface.co/docs/transformers/en/index) to replace closed Gemini model with open models such as [LLaMA2](https://llama.meta.com/), [Gemma](https://blog.google/technology/developers/gemma-open-models/), [Mistral](https://mistral.ai/), etc.,

## Acknowledgments

This is a project built during the Gemini sprint held by Google's ML Developer Programs team. We are thankful to be granted good amount of GCP credits to finish up this project.

