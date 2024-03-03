"""
Model management and generation routines for the toxicity research project.
"""

# imports
from pathlib import Path

# packages
from transformers import pipeline, Pipeline

# set local relative model path
MODEL_PATH = (Path(__file__).parent.parent / "models").resolve().absolute()


def load_model_kl3m_170m() -> Pipeline:
    """
    Load the KL-3M 170M model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation", model=str(MODEL_PATH / "kl3m-170m"), device_map="auto"
    )


def load_model_kl3m_1_7b() -> Pipeline:
    """
    Load the KL-3M 1.7B model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation", model=str(MODEL_PATH / "kl3m-1.7b"), device_map="auto"
    )


# methods to load models in specific configurations
def load_model_gpt2() -> Pipeline:
    """
    Load the GPT-2 model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline("text-generation", model="openai-community/gpt2", device_map="auto")


def load_model_gpt2_xl() -> Pipeline:
    """
    Load the GPT-2 Medium model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation", model="openai-community/gpt2-xl", device_map="auto"
    )


def load_model_pythia_160m() -> Pipeline:
    """
    Load the Pythia 160M model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation", model="EleutherAI/pythia-160m", device_map="auto"
    )


def load_model_pythia_1_4b() -> Pipeline:
    """
    Load the Pythia 1.4B model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation", model="EleutherAI/pythia-1.4b", device_map="auto"
    )


def load_model_gemma_2b() -> Pipeline:
    """
    Load the GEMMA 2B model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline("text-generation", model="google/gemma-2b", device_map="auto")


def load_model_tinyllama() -> Pipeline:
    """
    Load the TinyLlama 1.1B model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        device_map="auto",
    )


def load_model_stablelm2_1_6b() -> Pipeline:
    """
    Load the StableLM 2.1 6B model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation",
        model="stabilityai/stablelm-2-1_6b",
        device_map="auto",
        trust_remote_code=True,
    )


def load_model_phi2() -> Pipeline:
    """
    Load the Phi2 model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation",
        model="microsoft/phi-2",
        device_map="auto",
        trust_remote_code=True,
    )


def load_model_mistral_7b() -> Pipeline:
    """
    Load the Mistral 7B model for text generation.

    Returns:
        A pipeline for text generation.
    """
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-v0.1",
        device_map="auto",
        trust_remote_code=True,
    )


def generate_responses(
    model_pipeline: Pipeline,
    prompts: list[str],
    max_new_tokens: int = 50,
    temperature: float = 0.5,
    num_per_prompt: int = 3,
) -> list[dict]:
    """
    Generate responses to a list of prompts using the specified model pipeline.

    Args:
        model_pipeline: The model pipeline to use for text generation.
        prompts: The list of prompts to generate responses for.
        max_new_tokens: The maximum number of tokens to generate for each response.
        temperature: The temperature to use for sampling.
        num_per_prompt: The number of responses to generate for each prompt.

    Returns:
        A list of generated responses.
    """
    # generate responses
    responses = []
    for i, generation in enumerate(
        model_pipeline(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_per_prompt,
        )
    ):
        for response in generation:
            responses.append(
                {
                    "prompt": prompts[i],
                    "response": response["generated_text"],
                }
            )

    # extract the generated text
    return responses


# mapping from model names to loading functions
MODEL_LOADERS = {
    "kl3m-170m": load_model_kl3m_170m,
    "kl3m-1.7b": load_model_kl3m_1_7b,
    "gpt2": load_model_gpt2,
    "gpt2-xl": load_model_gpt2_xl,
    "pythia-160m": load_model_pythia_160m,
    "pythia-1.4b": load_model_pythia_1_4b,
    "gemma-2b": load_model_gemma_2b,
    "tinyllama": load_model_tinyllama,
    "stablelm2-1.6b": load_model_stablelm2_1_6b,
    "phi-2": load_model_phi2,
    "mistral-7b": load_model_mistral_7b,
}
