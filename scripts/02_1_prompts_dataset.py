import os
from loguru import logger
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import wandb
from llm_prompt import  REWRITE_PROMPTS
from sklearn.cluster import KMeans

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "../input")
INPUT_DATASET_TYPE = "rewritten_texts"
NUMBER_CLUSTERS = 10
SEED = 42
KEEP_COLUMNS = ["original_text", "rewritten_text", "rewrite_prompt", "source"]


def main() -> None:
    version = "downloaded"
    run = wandb.init(job_type="generate_prompts")
    dataset_name = f"v-{version}"
    artifact = run.use_artifact(f"{dataset_name}-{INPUT_DATASET_TYPE}:latest", type=INPUT_DATASET_TYPE)
    datadir = artifact.download(f"./artifacts/{INPUT_DATASET_TYPE}/{dataset_name}")
    dataset_dict = load_from_disk(datadir)
    df = pd.concat([dataset_dict[key].to_pandas() for key in dataset_dict.keys()], ignore_index=True)
    prompt_df = df[['source', 'rewrite_prompt']].drop_duplicates().reset_index(drop=True)
    all_custom_prompts_df = []
    for key, prompts in REWRITE_PROMPTS.items():
        custom_prompt_df = pd.DataFrame({"rewrite_prompt": prompts})
        custom_prompt_df['source'] = key
        all_custom_prompts_df.append(custom_prompt_df)
    all_custom_prompts_df = pd.concat(all_custom_prompts_df, ignore_index=True)
    prompt_df = pd.concat([prompt_df, all_custom_prompts_df], ignore_index=True).drop_duplicates().reset_index(drop=True)
    logger.info(f"Number of prompts: {len(prompt_df)}")
    
    model = SentenceTransformer("sentence-transformers/sentence-t5-base").to("cuda")
    embeddings = model.encode(prompt_df["rewrite_prompt"].tolist(), batch_size=64, show_progress_bar=True)
    prompt_df["embeddings"] = embeddings.tolist()
    kmeans = KMeans(n_clusters=NUMBER_CLUSTERS, random_state=SEED)
    kmeans.fit(embeddings)
    prompt_df["cluster"] = kmeans.labels_
    prompt_df.to_parquet(f"{INPUT_DATA_DIR}/prompts.parquet")
    np.save(f"{INPUT_DATA_DIR}/kmeans.npy", kmeans.cluster_centers_)
    prompt_artifact = wandb.Artifact("prompts", type="dataset")
    prompt_table = wandb.Table(dataframe=prompt_df)
    prompt_artifact.add_file(f"{INPUT_DATA_DIR}/prompts.parquet")
    prompt_artifact.add_file(f"{INPUT_DATA_DIR}/kmeans.npy")
    prompt_artifact.add(prompt_table, "prompts")
    run.log_artifact(prompt_artifact)
    run.finish()


if __name__ == "__main__":
    main()
