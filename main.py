import os
import modal
import uuid
import base64
import requests

from pydantic import BaseModel


app = modal.App('music-generator')

image = (
    modal.Image.debian_slim()
    .apt_install('git')
    .pip_install_from_requirements('requirements.txt')
    .run_commands(
        [
            'git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step',
            'cd /tmp/ACE-Step && pip install .',
        ]
    )
    .env({'HF_HOME': '/.cache/huggingface'})
    .add_local_python_source('prompts')
)


model_volume = modal.Volume.from_name('ace-step-models', create_if_missing=True)
hf_volume = modal.Volume.from_name('qwen-hf-cache', create_if_missing=True)

music_gen_secrets = modal.Secret.from_name('music-gen-secret')


class GenerateMusicResponse(BaseModel):
    audio_data: str
    image_data: str


@app.cls(
    image=image,
    gpu='L40S',
    volumes={'/models': model_volume, '/.cache/huggingface': hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15,
)
class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image
        import torch

        # Music Generation Model
        self.music_model = ACEStepPipeline(
            checkpoint_dir='/models',
            dtype='bfloat16',
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False,
        )

        # Large Language Model
        model_id = 'Qwen/Qwen2-7B-Instruct'
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype='auto',
            device_map='auto',
            cache_dir='/.cache/huggingface',
        )

        # Stable Diffusion Model (thumbnails)
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            'stabilityai/sdxl-turbo',
            torch_dtype=torch.float16,
            variant='fp16',
            cache_dir='/.cache/huggingface',
        )
        self.image_pipe.to('cuda')

    @modal.fastapi_endpoint(method='POST')
    def generate(self) -> GenerateMusicResponse:
        output_dir = '/tmp/outputs'
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filenames
        audio_filename = f'{uuid.uuid4()}.wav'
        image_filename = f'{uuid.uuid4()}.png'

        audio_path = os.path.join(output_dir, audio_filename)
        image_path = os.path.join(output_dir, image_filename)

        # Music generation prompt and lyrics
        music_prompt = 'electronic rap'
        lyrics = """[verse]
        Waves on the bass, pulsing in the speakers,
        Turn the dial up, we chasing six-figure features,
        Griding on the beats, codes in the creases,
        Digital hustler, midnight in sneakers.

        [chorus]
        Electro vibes, hearts beat with the hum,
        Urban legends ride, we ain't ever numb,
        Circuits sparking live, tapping on the drum,
        Living on the edge, never succumb.

        [verse]
        Synthesizers blaze, city lights a glow,
        Rhythm in the haze, moving with the flow,
        Swagger on stage, energy to blow,
        From the blocks to the booth, you already know.

        [bridge]
        Night's electric, streets full of dreams,
        Bass hits collective, bursting at seams,
        Hustle perspective, all in the schemes,
        Rise and reflective, ain't no in-betweens.

        [verse]
        Vibin' with the crew, sync in the wire,
        Got the dance moves, fire in the attire,
        Rhythm and blues, soul's our supplier,
        Run the digital zoo, higher and higher.

        [chorus]
        Electro vibes, hearts beat with the hum,
        Urban legends ride, we ain't ever numb,
        Circuits sparking live, tapping on the drum,
        Living on the edge, never succumb."""

        # Generate music
        self.music_model(
            prompt=music_prompt,
            lyrics=lyrics,
            audio_duration=180,
            infer_step=60,
            guidance_scale=15,
            save_path=audio_path,
        )

        # Generate album cover image
        image_prompt = f'{music_prompt} album cover, futuristic cyberpunk style, neon lights, urban cityscape, digital art, high quality'

        generated_image = self.image_pipe(
            prompt=image_prompt,
            num_inference_steps=4,  # SDXL-Turbo works well with 4 steps
            guidance_scale=0.0,  # SDXL-Turbo doesn't need guidance
        ).images[0]

        # Save the generated image
        generated_image.save(image_path)

        # Read and encode audio file
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # Read and encode image file
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Clean up temporary files
        os.remove(audio_path)
        os.remove(image_path)

        return GenerateMusicResponse(audio_data=audio_b64, image_data=image_b64)


@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate.get_web_url()

    response = requests.post(endpoint_url)
    response.raise_for_status()
    result = GenerateMusicResponse(**response.json())

    # Save audio file
    audio_data = base64.b64decode(result.audio_data)
    audio_output_path = 'ai-music.wav'
    with open(audio_output_path, 'wb') as f:
        f.write(audio_data)

    # Save image file
    image_data = base64.b64decode(result.image_data)
    image_output_path = 'ai-album-cover.png'
    with open(image_output_path, 'wb') as f:
        f.write(image_data)

    print(f'Generated music saved to: {audio_output_path}')
    print(f'Generated album cover saved to: {image_output_path}')
