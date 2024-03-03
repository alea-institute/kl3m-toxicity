"""
Output randomly generated prompts to an experiment ID.
"""

# imports
import argparse
import json
from pathlib import Path

# packages
import numpy

# project imports
from toxicity_research.data import INPUT_DATA_PATH, OUTPUT_DATA_PATH, get_all_prompts

# main method
if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Generate prompts for an experiment.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="001",
        help="The dataset ID to use for prompts.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="The number of prompts to generate.",
    )
    args = parser.parse_args()

    # load the prompts
    prompts = get_all_prompts()

    # select a random sample of prompts
    random_prompts = numpy.random.choice(prompts, size=args.num_prompts, replace=False)

    # id string is three-digit zero-padded or just str
    try:
        dataset_id = f"{int(args.dataset_id):03d}"
    except ValueError:
        dataset_id = args.dataset_id

    # write the prompts to the output path with three-digit zero-padded numbers
    output_path = INPUT_DATA_PATH / f"prompts-{dataset_id}.txt"
    with open(output_path, "wt", encoding="utf-8") as output_file:
        for prompt in random_prompts:
            output_file.write(prompt + "\n")
    print(f"Wrote {args.num_prompts} prompts to {output_path}.")
