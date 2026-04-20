# Similarity-Based Delegate Recommender

## Overview
This repository provides the implementation to compare different similiarity metrics as basis for a similarity-based recommender for delegate selection in DAOs, as described in the paper <TODO>. We collect historical voting behavior (votes cast by delegates on proposals) and use it to evaluate different similarity metrics based on their ability to identify delegates with similar voting patterns.

## Method Summary
The methodology consists of two stages:

1. **Data acquisition and transformation**
   - Implemented in **`prepare_data.py`**
   - Fetch proposals and votes for DAO using the Snapshot GraphQL API.
   - Identify balanced proposals (proposals where the winning choice had a voting power share below `0.7`).
   - Build a dense matrix (delegate × proposal; categorical vote label per proposal), obtained by pruning proposals repeatedly from the sparse voting data collected. Later used for evaluation.
   - Report insights about the collected data.

2. **Recommendation and evaluation**
   - Implemented in **`perform_evaluation.py`**
   - For each delegate in the dense matrix, the most similar delegate is identified and recommended using each similiarity metric.
   - The quality of these recommendations is evaluated on the evaluation set (proposals not present in the dense matrix). A further evaluation on only the balanced proposals within the evaluation set is also performed. 
   - The evaluation is carried out using the following evaluation criteria:
     - **Accuracy**: Fraction of shared proposals between two voters from the evaluation set, acting as delegate and delegator, where the delegate’s vote matches the delegator’s vote.
     - **Outperformance Ratio**: Fraction of recommendations in which the accuracy of the recommendation outperforms the accuracy of the random delegation baseline.
     - **Underperformance Error**: Mean absolute error between accuracy of the recommended delegate and the random delegation baseline when the recommended delegate's accuracy is lower.
     - **Perfect Delegate Recall**: Fraction of recommendations where the recommended delegate is a perfect delegate (delegate with accuracy=1), provided that a perfect delegate exists.


## Repository Structure
- **`prepare_data.py`**
  - Runs data fetching, transformation and provides insights about the data.
- **`perform_evaluation.py`**
  - Runs delegate recommendation for each similarity metric and provides evaluation results.
- **`utils.py`**
  - Contains helper functions for data fetching, transformation, similarity metrics, and evaluation.
- **`cfgs/*.yaml`**
  - DAO-specific configuration (DAO name, Snapshot space id, density threshold `tau`, and vote-label vectorization mapping for cosine similarity).
- **`data/`**
  - Snapshot data (proposals, votes, and dense matrix).

## Installation
Python dependencies are listed in `requirements.txt`.
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Configuration
Experiments for different DAOs are configured via YAML files in `cfgs/`. Each config includes:

- `name`: DAO name (used in input/output filenames)
- `space`: Snapshot space name (e.g., `uniswapgovernance.eth`)
- `limit_to_50k`: whether to stop vote fetching after 50k votes (used for high-volume spaces like Stargate DAO)
- `tau`: Number of proposals in generated dense matrix
- `vectorized_labels_mapping`: Mapping from proposal choice strings to numeric values (used for cosine similarity encoding)

### Data Preparation
The script `prepare_data.py` requires the path of the configuration file as an argument and supports:
1. The option to fetch data from Snapshot instead of using locally stored JSON files (eg. from a previous run) using the --fetch_data flag.
2. The generation of the dense matrix using the --create_dense_matrix flag.
3. Providing insights about the existing data.

#### Example usage for Uniswap:
Fetch and build dense matrix:
```bash
python prepare_data.py --config="cfgs/uniswap.yaml" --fetch_data --create_dense_matrix
```

Use cached data and rebuild dense matrix (eg. to try out different tau values):
```bash
python prepare_data.py --config="cfgs/uniswap.yaml" --create_dense_matrix
```

Only provide insights about local data:
```bash
python prepare_data.py --config="cfgs/uniswap.yaml"
```

#### Output files
The script writes the following files to `data/` (`name` is taken from the DAO name in the config):
- `snapshot_proposals_<name>.json`
- `all_votes_snapshot_<name>.json`
- `recommendation_dataset_matrix_<name>.json`

### Running Evaluation
After the dense matrix has been created at `data/recommendation_dataset_matrix_<name>.json`, run:

```bash
python perform_evaluation.py --config cfgs/uniswap.yaml
```
Additionally the results can be visualized using box plots by adding the `--visualize_results` flag:

```bash
python perform_evaluation.py --config cfgs/uniswap.yaml --visualize_results
```

## Practical Considerations
- **Network dependence**: Fetching data uses Snapshot’s GraphQL endpoint (`https://hub.snapshot.org/graphql`). Runs may be slow for large spaces, so it is recommended to use cached files after the first fetch.
- **`tau` sensitivity**: The dense matrix generation depends strongly on `tau`. Larger `tau` results in fewer delegates but more proposals in the dense matrix. The value should be experimentally determined for each DAO to balance delegate count and proposal coverage necessary for meaningful evaluation.

## Reproducing results from the paper
To reproduce the results from the paper use the collected data from the Zenodo repository <TODO> and run the evaluation script with the corresponding configuration file.
Example:
```bash
python perform_evaluation.py --config cfgs/uniswap.yaml
```
