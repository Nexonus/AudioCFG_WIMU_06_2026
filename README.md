# AudioCFG
WIMU Project for CFG with Null Embeddings on an audio-generating stable diffusion model - Warsaw's University of Technology, Winter 2026

PROJECT DESIGN PROPOSAL:

Project name: AudioCFG

Github repository:  

Project topic: “What guidance does to generated audio: the guidance scale, the null embedding, diversity, copying and what judges prefer”.

What needs to be done:
A stable-diffusion model, that will use text prompts to generate audio or sound effects. Eg. “generate an effect of birds flying by on a windy autumn day.” This would yield a result audio wave in the dashboard.
CFG Slider (Class-Free Guidance is a measure of how much creative freedom should we give a model, low values may cause problems as a model will not fully follow the prompt’s instructions. Too high values will make the neural network deterministic and sound highly synthetic. Our objective is to implement such hyperparameter and search for optimal values that will produce realistic audio following user’s instructions.

Tech stack, Tools proposals:
Python, Py-Torch
Stable Audio Open: https://huggingface.co/stabilityai/stable-audio-open-1.0
Gradio: https://gradio.app/guides/quickstart
Evaluation libraries (CLAP, FAD) 
https://github.com/gudgud96/frechet-audio-distance
https://github.com/LAION-AI/CLAP

Evaluation metrics:
CLAP (How much the generated audio is related to the input prompt, above 0.4 very good, 0.25 good, below 0.15 very bad).
FAD (Sound’s quality and how much realistic the output audio is). 
MOS (Mean Opinion Score - human-based evaluation, for example 10 listeners rating how much realistic the audio file is).
