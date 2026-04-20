from pandas._config import config
from utils import *
import argparse
import pandas as pd
from tqdm import tqdm
# loading cfg
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
parser.add_argument("--fetch_data", action="store_true", help="Fetch data instead of using local files")
parser.add_argument("--create_dense_matrix", action="store_true", help="Create dense matrix using voting data")
args = parser.parse_args()
cfg = load_config(args.config)

if args.fetch_data:
    # get proposal data and save it.
    proposals_df = fetch_proposals(cfg["space"])
    # saving proposals as json
    proposals_df.to_json(f'data/snapshot_proposals_{cfg["name"]}.json', orient='records', indent=4)
    print(f"Saved proposals to data/snapshot_proposals_{cfg['name']}.json")
else:
    proposals_df = pd.read_json(f'data/snapshot_proposals_{cfg["name"]}.json')

print("Number of proposals:", len(proposals_df))

#defining controversial proposals
mask = proposals_df['percentage_scores'].apply(lambda scores: max(scores) < 0.7)
controversial_proposals = proposals_df[mask]
print("Number of controversial proposals:", len(controversial_proposals))

# exploring types of proposals
print("Proposal types for all proposals:",get_proposal_type_count(proposals_df))
print("Proposal types for controversial proposals:",get_proposal_type_count(controversial_proposals))


if args.fetch_data:
    # getting votes for each proposal
    voting_data = []
    for _, proposal in tqdm(proposals_df.iterrows(), total=len(proposals_df), desc="Fetching votes for proposals"):
        votes_for_proposal = fetch_votes(proposal['id'], cfg["limit_to_50k"])
        for vote in votes_for_proposal:
            vote['proposal_id'] = proposal['id']
            voting_data.append(vote)
    voting_df = pd.DataFrame(voting_data)

    # saving voter data
    voting_df.to_json(f'data/all_votes_snapshot_{cfg["name"]}.json', orient='records', indent=4)
    print(f"Saved voting data to data/all_votes_snapshot_{cfg['name']}.json")

else:
    # get voting data
    with open(f"data/all_votes_snapshot_{cfg['name']}.json", "r") as f:
        data = json.load(f)
    voting_df = pd.DataFrame(data)

# Get participation stats of delegates

# setting participation value
voting_df['participated'] = 1
# creating voter proposal matrix
voter_proposal_matrix = voting_df.pivot_table(
    index='voter',
    columns='proposal_id',
    values='participated',
    fill_value=0
)
print("Total number of delegates:", len(voter_proposal_matrix))
print("Total number of proposals:", len(voter_proposal_matrix.columns))
#getting number of active delegates
participation_rate = voter_proposal_matrix.sum(axis=1) / voter_proposal_matrix.shape[1]
active_delegates = voter_proposal_matrix[participation_rate >= 0.5]
print("Number of active delegates:", len(active_delegates))

if args.create_dense_matrix:
    # Creating dense matrix from sparse voter proposal matrix
    recommendation_dataset = build_dataset(
    voter_proposal_matrix.loc[:, voter_proposal_matrix.columns.isin(set(controversial_proposals["id"]))],
    cfg["tau"]
    )
    recommendation_dataset = recommendation_dataset.drop('voter_participation', axis = 1)
    proposal_choices_dict = controversial_proposals.set_index('id')['choices'].to_dict()
    voting_df['choice_value'] = voting_df.apply(map_choice_value, axis=1)
    voter_proposal_matrix = voting_df.pivot_table(
        index='voter',
        columns='proposal_id',
        values='choice_value',
        aggfunc='first',
        fill_value="NaN"
    )
    recommendation_dataset_matrix = voter_proposal_matrix.loc[voter_proposal_matrix.index.isin(recommendation_dataset.index)]
    recommendation_dataset_matrix = recommendation_dataset_matrix.drop([col for col in recommendation_dataset_matrix.columns if col not in recommendation_dataset.columns], axis = 1)
    recommendation_dataset_matrix[recommendation_dataset_matrix.columns] = recommendation_dataset_matrix[recommendation_dataset_matrix.columns].astype(int)
    recommendation_dataset_matrix.to_json(f'data/recommendation_dataset_matrix_{cfg["name"]}.json', orient='table', indent=4)
    print("Dense matrix shape (delegates,proposals):", recommendation_dataset_matrix.shape)
    print(f"Saved Dense Matrix to data/recommendation_dataset_matrix_{cfg['name']}.json")
else:
    recommendation_dataset_matrix = pd.read_json(f'data/recommendation_dataset_matrix_{cfg["name"]}.json', orient='table')
    print("Dense matrix shape (delegates,proposals):", recommendation_dataset_matrix.shape)
