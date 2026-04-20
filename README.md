# Similarity-Based Delegate Recommender

This repository provides the implementation to compare the performance of various similarity metrics as basis for a similarity-based recommender for delegate selection in DAOs. The code is complementary to the paper "Kaushik, A., Droll, J. (2026). Who Should Represent Me? Similarity-Based Recommender Systems for Vote Delegation in DAOs. IEEE International Conference on Blockchain and Cryptocurrency (ICBC). Brisbane, Australia."

## Overview
The methodology is described in detail in Section 3 of the above mentioned paper and consists of the following two stages:

1. **Data acquisition and transformation**
   - Fetch proposals and votes for a DAO using the [Snapshot GraphQL API](https://docs.snapshot.box/tools/api).
   - Identify balanced proposals (proposals where the winning choice had a voting power share below `0.7`).
   - Build a dense matrix (delegate × proposal; categorical vote label per proposal), obtained by pruning proposals repeatedly from the sparse voting data collected. Later used for evaluation.
   - Report statistics about the collected data.

2. **Recommendation and evaluation**
   - For each delegate in the dense matrix, the most similar delegate is identified and recommended for each similarity metric.
   - The quality of these recommendations is evaluated on the evaluation set (proposals not present in the dense matrix). A further evaluation on only the balanced proposals within the evaluation set is also performed. 
   - The evaluation is carried out using the following evaluation criteria:
     - **Accuracy**: Fraction of shared proposals between two voters from the evaluation set, acting as delegate and delegator, where the delegate’s vote matches the delegator’s vote.
     - **Outperformance Ratio**: Fraction of recommendations in which the accuracy of the recommendation outperforms the accuracy of the random delegation baseline.
     - **Underperformance Error**: Mean absolute error between accuracy of the recommended delegate and the random delegation baseline when the recommended delegate's accuracy is lower.
     - **Perfect Delegate Recall**: Fraction of recommendations where the recommended delegate is a perfect delegate (delegate with accuracy=1), provided that a perfect delegate exists.


## Repository Structure
- **`prepare_data.py`**
  - Implements data fetching and transformation, and provides insights about the data.
- **`perform_evaluation.py`**
  - Implements delegate recommendation and performance evaluation for each similarity metric.
- **`utils.py`**
  - Contains helper functions for data fetching, transformation, similarity metrics, and evaluation.
- **`cfgs/*.yaml`**
  - DAO-specific configurations (DAO name, Snapshot space id, density threshold `tau`, and vote-label vectorization mapping for cosine similarity).
- **`data/`**
  - Snapshot data as well as artifacts (proposals, votes, and dense matrix).

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
- `limit_to_50k`: Whether to stop vote fetching after 50k votes (used for high-volume spaces like Stargate DAO)
- `tau`: Number of proposals in generated dense matrix
- `vectorized_labels_mapping`: Mapping from proposal choice strings to numeric values (used for cosine similarity encoding)

### Data Preparation
The script `prepare_data.py` requires the path of the configuration file as an argument and supports:
1. The option to fetch data from Snapshot instead of using locally stored JSON files (eg. from a previous run) using the `--fetch_data` flag.
2. The generation of the dense matrix using the `--create_dense_matrix` flag.
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
- **`tau` sensitivity**: The dense matrix generation depends strongly on `tau`. Larger `tau` values result in fewer delegates but more proposals in the dense matrix. The value should be experimentally determined for each DAO to balance delegate count and proposal coverage necessary for meaningful evaluation.

## Reproducing results from the paper
To reproduce the results from the paper use the collected data from the Zenodo repository <TODO> and run the evaluation script with the corresponding configuration file.
Example:
```bash
python perform_evaluation.py --config cfgs/uniswap.yaml
```

## License
MIT License (see `LICENSE`).