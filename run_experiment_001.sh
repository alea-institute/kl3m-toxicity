MODEL_LIST="gpt2,gpt2-xl,pythia-160m,pythia-1.4b,tinyllama"
MODEL_LIST="$MODEL_LIST,phi-2,stablelm2-1.6b,kl3m-170m,kl3m-1.7b"

PYTHONPATH=. poetry run python3 \
  toxicity_research/cli/generate_samples.py \
  --dataset-id 001 \
  "$MODEL_LIST" \
  data/output/output-001.jsonl.gz

PYTHONPATH=. poetry run python3 \
  toxicity_research/cli/score_samples.py \
  --dataset-id 001

PYTHONPATH=. poetry run python3 \
  toxicity_research/cli/analyze_samples.py \
  --dataset-id 001
