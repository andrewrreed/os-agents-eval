import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets


def build_dataset() -> Dataset:
    """
    Builds a dataset by combining examples from the HotpotQA and GAIA datasets.

    Returns:
        dataset (Dataset): The combined dataset with added indices.
    """
    # hotpotqa dataset
    # let's sample a few examples from each level (of difficulty) and type (comparion or bridge)
    hotpotqa_dataset = load_dataset("hotpot_qa", "distractor")
    hotpotqa_dataset.set_format("pandas")
    dataset_df = hotpotqa_dataset["train"][:]
    sample_indicies = (
        dataset_df.groupby(["level", "type"]).sample(4, random_state=10).index.values
    )
    hotpotqa_dataset.reset_format()
    hotpotqa_dataset = hotpotqa_dataset["train"].select(sample_indicies)
    task_column = [f"HotpotQA-{level}" for level in hotpotqa_dataset["level"]]
    hotpotqa_dataset = hotpotqa_dataset.add_column("task", task_column).select_columns(
        ["question", "answer", "task"]
    )

    gaia_dataset = load_dataset("gaia-benchmark/GAIA", "2023_level1")["validation"]

    # gaia dataset
    # we'll manually select "easy" examples that can be solved with search and calculator tools
    gaia_dataset.set_format("pandas")
    gaia_dataset_df = gaia_dataset[:]
    gaia_dataset_df["number_of_steps"] = gaia_dataset_df["Annotator Metadata"].apply(
        lambda row: int(row["Number of steps"])
    )
    gaia_dataset_df["tools_used"] = gaia_dataset_df["Annotator Metadata"].apply(
        lambda row: row["Tools"]
    )
    gaia_dataset_df = gaia_dataset_df.loc[
        ~gaia_dataset_df["tools_used"]
        .str.lower()
        .str.contains(
            "pdf|excel|image|video|parsing|audio|word|file|speech|viewer|markdown|python|editor"
        )
    ]
    selected_indicies = [1, 18, 23, 29, 39, 42, 47, 49, 50, 52]
    gaia_dataset = gaia_dataset.rename_columns(
        {"Question": "question", "Final answer": "answer"}
    ).select_columns(["question", "answer"])
    gaia_dataset.reset_format()
    gaia_dataset = gaia_dataset.select(selected_indicies)

    task_column = ["GAIA"] * len(gaia_dataset)
    gaia_dataset = gaia_dataset.add_column("task", task_column)

    # combine and add id's
    dataset = concatenate_datasets([hotpotqa_dataset, gaia_dataset])

    def add_index(example, idx):
        return {**example, "id": idx}

    return dataset.map(add_index, with_indices=True)
