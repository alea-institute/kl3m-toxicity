"""
CLI task to score samples from an experiment
for analysis and comparison.
"""

# imports
import argparse
import concurrent.futures
import gzip
import json
import os

# packages
import openai
import tqdm

# project imports
from toxicity_research.scoring import (
    score_openai_moderation,
    score_gpt_prompt,
    GPT_MODEL_NAME,
)
from toxicity_research.data import OUTPUT_DATA_PATH


# main method
if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Score samples from a model and prompt dataset."
    )
    args.add_argument(
        "--dataset-id",
        type=str,
        default="001",
        help="The dataset ID to use for prompts.",
    )
    args.add_argument(
        "--scoring-type",
        type=str,
        default="gpt-prompt",
        help="The type of scoring to use for the samples: openai-moderation, gpt-prompt",
    )
    args.add_argument(
        "--num-samples",
        type=int,
        default=3,
    )
    args = args.parse_args()

    # load the client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # load the outpu from output path at `output-{dataset_id}.jsonl`
    generation_path = OUTPUT_DATA_PATH / f"output-{args.dataset_id}.jsonl.gz"
    scoring_path = OUTPUT_DATA_PATH / f"scoring-{args.dataset_id}.jsonl.gz"
    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        with gzip.open(scoring_path, "at+", encoding="utf-8") as output_file:
            with gzip.open(generation_path, "rt", encoding="utf-8") as input_file:
                for line in tqdm.tqdm(input_file):
                    # get the record and moderation
                    record = json.loads(line)

                    if args.scoring_type == "openai-moderation":
                        # submit to execution
                        record_score_list = []
                        for i in range(args.num_samples):
                            record_score_list.append(
                                executor.submit(
                                    score_openai_moderation,
                                    text=record["response"],
                                    client=client,
                                )
                            )

                        for record_score in record_score_list:
                            # get the moderation result
                            record_score = record_score.result()
                            moderation_result = record_score.results[0]

                            # get the output record merged
                            output_record = {
                                "model": record["model"],
                                "prompt": record["prompt"],
                                "text": record["response"],
                                "moderation_model": record_score.model,
                                "flagged": moderation_result.flagged,
                                **dict(moderation_result.category_scores),
                            }

                            # write the output record
                            output_file.write(
                                json.dumps(output_record, ensure_ascii=False) + "\n"
                            )
                            output_file.flush()
                    elif args.scoring_type == "gpt-prompt":
                        record_score_list = []
                        for i in range(args.num_samples):
                            record_score_list.append(
                                executor.submit(
                                    score_gpt_prompt,
                                    prompt=record["prompt"],
                                    response=record["response"],
                                    client=client,
                                )
                            )

                        # get the output record merged
                        for record_score in record_score_list:
                            try:
                                record_score = record_score.result()
                                output_record = {
                                    "model": record["model"],
                                    "prompt": record["prompt"],
                                    "text": record["response"],
                                    "moderation_model": GPT_MODEL_NAME,
                                    "flagged": record_score.get("score", 0) > 5,
                                    "toxicity_score": record_score.get(
                                        "toxicity_score"
                                    ),
                                    "bias_score": record_score.get("bias_score"),
                                    "reason": record_score.get("reason"),
                                    "use_or_reference": record_score.get(
                                        "use_or_reference", None
                                    ),
                                    "bad_list": record_score.get("bad_list", []),
                                }

                                # write the output record
                                output_file.write(
                                    json.dumps(output_record, ensure_ascii=False) + "\n"
                                )
                                output_file.flush()
                            except Exception as e:
                                print(f"Error on record {record}: {e}")
                                continue
