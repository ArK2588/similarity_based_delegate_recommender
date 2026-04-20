from utils import *
import argparse
import pandas as pd
import numpy as np

# loading cfg
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
parser.add_argument("--visualize_results", action="store_true", help="Visualize results using matplotlib")
args = parser.parse_args()
cfg = load_config(args.config)

# read dense matrix
recommendation_dataset = pd.read_json(f'data/recommendation_dataset_matrix_{cfg["name"]}.json', orient='table')

# get proposal choices
choices = get_proposal_choices(cfg["name"])

# get voting data
with open(f"data/all_votes_snapshot_{cfg['name']}.json", "r") as f:
    data = json.load(f)
voting_df = pd.DataFrame(data)
voting_df['choice_value'] = voting_df.apply(map_choice_value, axis=1)
freq_dist = get_freq_dist_vp_matrix(recommendation_dataset)

# get vectorized labels mapping
vectorized_labels_mapping = cfg["vectorized_labels_mapping"]
# getting transformed dataframe for cosine similarity calculation
transformed_df = encode_labels(recommendation_dataset,choices,vectorized_labels_mapping)

# performing the recommendation
recommended_delegates = {'cosine_similarity':[], 'overlap': [],'eskin': [],'iof':[], 'of': [], 'lin': [], 'goodall1': [], 'goodall2': [], 'goodall3': [], 'goodall4': []}
for voter_id in recommendation_dataset.index:
    # storing results of cosine similiarity
    for metric in recommended_delegates.keys():
        if metric == "cosine_similarity":
            top_delegate = recommend_delegates(voter_id,transformed_df,metric,freq_dist).iloc[0,0]
        else:
            top_delegate = recommend_delegates(voter_id,recommendation_dataset,metric,freq_dist).iloc[0,0]
        recommended_delegates[metric].append((voter_id,top_delegate))

# evaluate on non-controversial proposals
print(f"\n{'='*50}")
print("Evaluation across all proposals in the evaluation set")
print(f"\n{'='*50}")
voting_df_non_cont = voting_df[~voting_df.proposal_id.isin(recommendation_dataset.columns)]
evaluate_similarity_metrics(voting_df_non_cont, recommendation_dataset, recommended_delegates, cfg, args.visualize_results)
 
print(f"\n{'='*50}")
print("Evaluation across balanced proposals in the evaluation set")
print(f"\n{'='*50}")
 
# Evaluate on controversial proposals
proposals = pd.read_json(f'data/snapshot_proposals_{cfg["name"]}.json', orient='records')
mask = proposals['percentage_scores'].apply(lambda scores: max(scores) < 0.7)
controversial_proposals = proposals[mask]
 
# Get voting data for controversial proposals
voting_df_cont = voting_df[voting_df.proposal_id.isin(controversial_proposals["id"])]
voting_df_cont = voting_df_cont[~voting_df_cont.proposal_id.isin(recommendation_dataset.columns)]
proposal_choices_dict = proposals.set_index('id')['choices'].to_dict()
voting_df_cont['choice_value'] = voting_df_cont.apply(map_choice_value, axis=1, result_type="reduce")
evaluate_similarity_metrics(voting_df_cont, recommendation_dataset, recommended_delegates, cfg, args.visualize_results)

# Evaluate participation boost
voting_df['participated'] = 1
# Creating voter proposal matrix
voter_proposal_matrix = voting_df.pivot_table(
    index='voter',
    columns='proposal_id',
    values='participated',
    fill_value=0
)
voter_proposal_matrix = enhance_voter_proposal_matrix(voter_proposal_matrix)
# Create participation statistics table
sparse_stats = voter_proposal_matrix["voter_participation"].describe()
dense_stats = voter_proposal_matrix[voter_proposal_matrix.index.isin(recommendation_dataset.index)]['voter_participation'].describe()
participation_table = pd.DataFrame({
    'Sparse Matrix': sparse_stats,
    'Dense Matrix': dense_stats
})
print("Delegate Participation Statistics")
print(participation_table.round(4).to_string())