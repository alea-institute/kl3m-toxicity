"""
CLI task to generate samples from models and prompt data
to use for scoring and analysis.
"""

# imports
import argparse
import gzip
import json
from pathlib import Path

# packages
import tqdm

# project imports
from toxicity_research.data import load_text_prompts, load_json_prompts
from toxicity_research.models import MODEL_LOADERS, generate_responses


# main method
if __name__ == "__main__":
    # setup the arg parser for this:
    # - prompt dataset id (default: 001)
    # - prompt dataset type (default: text)
    # - model name, required
    args = argparse.ArgumentParser(
        description="Generate samples from a model and prompt dataset."
    )
    args.add_argument(
        "--dataset-id",
        type=str,
        default="001",
        help="The dataset ID to use for prompts.",
    )
    args.add_argument(
        "--dataset-type",
        type=str,
        default="text",
        help="The dataset type to use for prompts.",
    )
    args.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="The maximum number of tokens to generate for each response.",
    )
    args.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="The temperature to use for sampling.",
    )
    args.add_argument(
        "--num-per-prompt",
        type=int,
        default=3,
        help="The number of responses to generate for each prompt.",
    )
    args.add_argument(
        "model_name",
        type=str,
        help="The name of the model to use for generation.",
    )
    args.add_argument(
        "output_path",
        type=str,
        help="The path to write the generated samples to.",
    )
    # parse the args
    args = args.parse_args()

    # split the model arg by comma
    model_list = args.model_name.split(",")

    for model_name in model_list:
        # load the model
        model_loader = MODEL_LOADERS[model_name]
        model = model_loader()

        # load the prompts
        if args.dataset_type == "text":
            prompts = load_text_prompts(args.dataset_id)
        elif args.dataset_type == "json":
            prompts = load_json_prompts(args.dataset_id)
        else:
            raise ValueError(f"Invalid dataset type: {args.dataset_type}")

        # generate the samples
        samples = generate_responses(
            model_pipeline=model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_per_prompt=args.num_per_prompt,
        )

        # open the output path as a+ utf-8
        with gzip.open(args.output_path, "at+", encoding="utf-8") as output_file:
            # write the samples to the file
            for sample in samples:
                record = {
                    "model": model_name,
                    "prompt": sample.get("prompt"),
                    "response": sample.get("response"),
                }
                output_file.write(json.dumps(record) + "\n")

        # clear the model/pipeline
        del model
        del model_loader
