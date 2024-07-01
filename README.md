https://text-to-image-hans.streamlit.app/

Using Streamlit, this webpage allows you to browse an image and upload it, in which it is processed by a Hugging Face model that creates a text prompt of what the image is about. With it, it then creates a short story with and audio of it - all generated using GenAI and LLMs. 

Hugging face models used:
* Image Captioning: Salesforce/blip-image-captioning-large
* Story Generation: openai-community/gpt2
* Audio Converter: espnet/kan-bayashi_ljspeech_vits
