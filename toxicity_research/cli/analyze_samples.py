"""
CLI task to analyze scored output samples
and generated statistics and summary data.
"""

# imports
import argparse

# packages
import pandas
import matplotlib.pyplot as plt

# project imports
from toxicity_research.data import OUTPUT_DATA_PATH, FIGURE_DATA_PATH


# set font
plt.rcParams["font.family"] = "Inter Tight"


def plot_bad_word_scores(
    data: pandas.DataFrame,
    dataset_id: str,
):
    """
    Plot the number of bad words/phrases by model.

    Args:
        data (pandas.DataFrame): The scoring data to analyze.
        dataset_id (str): The dataset ID to use for prompts.
    """

    # create the figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # get the group by score for bad_list_score
    # bad_list_scores = scoring_data.groupby("model")["bad_list_score"].sum().sort_values()
    # get the number of words in the text via split
    data["bad_list_score"] = 0.0
    for row_id, row in data.iterrows():
        num_words = len(row["text"].split())
        bad_words = 0
        for word in row["bad_list"]:
            bad_words += len(word.split())
        data.at[row_id, "bad_list_score"] = float(bad_words / num_words)

    # get the mean by model for bad_list_score
    bad_list_scores = (
        data.groupby("model")["bad_list_score"].sum()
        / data.groupby("model")["bad_list_score"].count()
    )
    bad_list_scores = bad_list_scores.sort_values(ascending=False)

    # plot horizontal bars and add tick font sizes
    bad_list_scores.plot(kind="barh", ax=ax)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)

    # label and remove legend
    plt.xlabel('"Bad" Word Rate', fontsize=20)
    plt.ylabel("Model", fontsize=20)
    plt.title(f'"Bad" Word Rate\n(Experiment {dataset_id})', fontsize=24)
    ax.legend().set_visible(False)

    # add a text box with more methodology explanation at the bottom
    text = """Full methodology and data, including prompts and toxicity scoring results, 
    are available on GitHub at: https://github.com/273v/kl3m-toxicity"""
    fig.text(0.5, 0.02, text, ha="center", fontsize=11, wrap=True)
    fig.savefig(FIGURE_DATA_PATH / f"bad-words-phrases-{args.dataset_id}.png")


def plot_toxicity_score(
    data: pandas.DataFrame,
    dataset_id: str,
    sort_by: str = "mean",
):
    """
    Plot the toxicity scores by model.

    Args:
        data (pandas.DataFrame): The scoring data to analyze.
        dataset_id (str): The dataset ID to use for prompts.
        sort_by (str): The method to use to sort the models. Default is "mean".
    """
    # if no score field, then check for toxicity_score and set it to score
    if "score" not in data.columns:
        if "toxicity_score" in data.columns:
            data["score"] = data["toxicity_score"]
        else:
            raise ValueError(
                "No score or toxicity_score field found in the scoring data."
            )

    # get the scores by model and sort
    if sort_by == "mean":
        model_scores = data.groupby("model")["score"].mean().sort_values()
    elif sort_by == "median":
        model_scores = data.groupby("model")["score"].median().sort_values()
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}")

    # get the raw data and re-order by model scores
    raw_data = data.groupby(["model", "score"]).size().unstack()
    raw_data = raw_data.loc[reversed(model_scores.index), :]

    # row-normalize and make percentages
    raw_data = raw_data.div(raw_data.sum(axis=1), axis=0) * 100.0

    # reverse the "RdYlGn" color map
    cmap = plt.get_cmap("RdYlGn")
    cmap = cmap.reversed()

    # set up the figure and horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 12))
    raw_data.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        cmap=cmap,
    )

    # label ticks and axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel("Percent of Responses by Toxicity Score", fontsize=20)
    plt.ylabel("Model", fontsize=20)
    plt.title(f"Model Toxicity Scores\n(Experiment {args.dataset_id})", fontsize=24)

    # remove legend
    ax.legend().set_visible(False)

    # add a text box with more methodology explanation at the bottom
    text = """Full methodology and data, including prompts and toxicity scoring results, 
        are available on GitHub at: https://github.com/273v/kl3m-toxicity"""
    fig.text(0.5, 0.02, text, ha="center", fontsize=12, wrap=True)

    # save the figure
    fig.savefig(FIGURE_DATA_PATH / f"toxicity-scores-{dataset_id}.png")


def plot_bias_score(
    data: pandas.DataFrame,
    dataset_id: str,
    sort_by: str = "mean",
):
    """
    Plot the toxicity scores by model.

    Args:
        data (pandas.DataFrame): The scoring data to analyze.
        dataset_id (str): The dataset ID to use for prompts.
        sort_by (str): The method to use to sort the models. Default is "mean".
    """
    # if no score field, then check for toxicity_score and set it to score
    if "bias_score" in data.columns:
        data["score"] = data["bias_score"]
    else:
        print("No bias_score field found in the scoring data.")
        return

    # get the scores by model and sort
    if sort_by == "mean":
        model_scores = data.groupby("model")["score"].mean().sort_values()
    elif sort_by == "median":
        model_scores = data.groupby("model")["score"].median().sort_values()
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}")

    # get the raw data and re-order by model scores
    raw_data = data.groupby(["model", "score"]).size().unstack()
    raw_data = raw_data.loc[reversed(model_scores.index), :]

    # row-normalize and make percentages
    raw_data = raw_data.div(raw_data.sum(axis=1), axis=0) * 100.0

    # reverse the "RdYlGn" color map
    cmap = plt.get_cmap("RdYlGn")
    cmap = cmap.reversed()

    # set up the figure and horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 12))
    raw_data.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        cmap=cmap,
    )

    # label ticks and axes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=12)
    plt.xlabel("Percent of Responses by Bias Score", fontsize=20)
    plt.ylabel("Model", fontsize=20)
    plt.title(f"Model Bias Scores\n(Experiment {args.dataset_id})", fontsize=24)

    # remove legend
    ax.legend().set_visible(False)

    # add a text box with more methodology explanation at the bottom
    text = """Full methodology and data, including prompts and bias scoring results, 
        are available on GitHub at: https://github.com/273v/kl3m-toxicity"""
    fig.text(0.5, 0.02, text, ha="center", fontsize=12, wrap=True)

    # save the figure
    fig.savefig(FIGURE_DATA_PATH / f"bias-scores-{dataset_id}.png")


if __name__ == "__main__":
    # parse the arguments
    parser = argparse.ArgumentParser(description="Analyze scored output samples.")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="003",
        help="The dataset ID to use for prompts.",
    )
    args = parser.parse_args()

    # load the scoring data
    scoring_path = OUTPUT_DATA_PATH / f"scoring-{args.dataset_id}.jsonl.gz"
    scoring_data = pandas.read_json(scoring_path, lines=True)

    # get the number of bad word/phrases by counting the length of the bad_list field
    scoring_data["bad_list_score"] = scoring_data["bad_list"].apply(len)

    # save as spreadsheet for easier consumption/review
    scoring_data.to_csv(
        OUTPUT_DATA_PATH / f"scoring-{args.dataset_id}.csv", encoding="utf-8"
    )

    # show the number of records per model
    model_counts = scoring_data["model"].value_counts()
    print("Counts:")
    print(model_counts)

    # plot the bad word scores
    plot_bad_word_scores(scoring_data, args.dataset_id)

    # plot the toxicity scores
    plot_toxicity_score(scoring_data, args.dataset_id)

    # plot the bias scores
    plot_bias_score(scoring_data, args.dataset_id)
