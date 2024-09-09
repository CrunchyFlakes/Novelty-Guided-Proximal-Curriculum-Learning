import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import argparse
from pathlib import PosixPath
import json
import os


def parse_result_info(result_infos: list[dict[str, float]], fill: bool) -> pd.DataFrame:
    df = pd.DataFrame(
        [result_info["score_history"] for result_info in result_infos]
    ).transpose()
    if fill:
        df = df.ffill()
    df = (
        df.add_prefix("Score")
        .reset_index()
        .rename(columns={"index": "Timestep"})
        .astype({"Timestep": int})
    )
    return df


def result_info_frame_to_long(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.wide_to_long(df, stubnames="Score", i="Timestep", j="Seed")
        .reset_index()
        .astype({"Seed": "category"})
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--comb_result", type=PosixPath, required=True)
    parser.add_argument("--prox_result", type=PosixPath, required=True)
    parser.add_argument("--nov_result", type=PosixPath, required=True)
    parser.add_argument("--vanilla_result", type=PosixPath, required=True)
    parser.add_argument("--context", type=str, required=True, help="Seaborn context")
    parser.add_argument("--output_dir", type=PosixPath, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.comb_result, "r") as result_file:
        comb_result = json.load(result_file)
        comb_result_info = result_info_frame_to_long(
            parse_result_info(comb_result, False)
        )
        comb_result_info_filled = result_info_frame_to_long(
            parse_result_info(comb_result, True)
        )
        comb_result_info["Approach"] = "Combined"
        comb_result_info_filled["Approach"] = "Combined"
    with open(args.prox_result, "r") as result_file:
        prox_result = json.load(result_file)
        prox_result_info = result_info_frame_to_long(
            parse_result_info(prox_result, False)
        )
        prox_result_info_filled = result_info_frame_to_long(
            parse_result_info(prox_result, True)
        )
        prox_result_info["Approach"] = "Proximal Curriculum"
        prox_result_info_filled["Approach"] = "Proximal Curriculum"
    with open(args.nov_result, "r") as result_file:
        nov_result = json.load(result_file)
        nov_result_info = result_info_frame_to_long(
            parse_result_info(nov_result, False)
        )
        nov_result_info_filled = result_info_frame_to_long(
            parse_result_info(nov_result, True)
        )
        nov_result_info["Approach"] = "State Novelty (RND)"
        nov_result_info_filled["Approach"] = "State Novelty (RND)"
    with open(args.vanilla_result, "r") as result_file:
        vanilla_result = json.load(result_file)
        vanilla_result_info = result_info_frame_to_long(
            parse_result_info(vanilla_result, False)
        )
        vanilla_result_info_filled = result_info_frame_to_long(
            parse_result_info(vanilla_result, True)
        )
        vanilla_result_info["Approach"] = "Vanilla"
        vanilla_result_info_filled["Approach"] = "Vanilla"

    result_infos = pd.concat(
        (comb_result_info, prox_result_info, nov_result_info, vanilla_result_info)
    ).reset_index()
    result_infos_filled = pd.concat(
        (
            comb_result_info_filled,
            prox_result_info_filled,
            nov_result_info_filled,
            vanilla_result_info_filled,
        )
    ).reset_index()

    matplotlib.use("qtAgg")
    sns.set_theme(context=args.context, style="darkgrid", rc={'figure.figsize': ((4, 2.5))})

    xlim = result_infos["Timestep"].max()

    # Results per Approach, full plot
    fig, ax = plt.subplots()
    sns.lineplot(data=result_infos_filled, x="Timestep", y="Score", hue="Approach")
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=xlim)
    if args.context == "talk":
        plt.xticks(rotation=45)
    # Remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    plt.tight_layout()
    plt.savefig(args.output_dir / f"results_per_approach_{args.context}.svg")

    # Results one plot per approach showing different seeds

    ## Combined
    fig, ax = plt.subplots()
    sns.lineplot(
        data=result_infos[result_infos["Approach"] == "Combined"],
        x="Timestep",
        y="Score",
        hue="Seed",
    )
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=xlim)
    if args.context == "talk":
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"results_combined_{args.context}.svg")

    ## Proximal Curriculum
    fig, ax = plt.subplots()
    sns.lineplot(
        data=result_infos[result_infos["Approach"] == "Proximal Curriculum"],
        x="Timestep",
        y="Score",
        hue="Seed",
    )
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=xlim)
    if args.context == "talk":
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"results_prox_{args.context}.svg")

    ## State Novelty
    fig, ax = plt.subplots()
    sns.lineplot(
        data=result_infos[result_infos["Approach"] == "State Novelty (RND)"],
        x="Timestep",
        y="Score",
        hue="Seed",
    )
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=xlim)
    if args.context == "talk":
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"results_nov_{args.context}.svg")

    ## Vanilla
    fig, ax = plt.subplots()
    sns.lineplot(
        data=result_infos[result_infos["Approach"] == "Vanilla"],
        x="Timestep",
        y="Score",
        hue="Seed",
    )
    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=xlim)
    if args.context == "talk":
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"results_vanilla_{args.context}.svg")
