import json
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import yaml
import numpy as np

# Setting snapshot graphql endpoint
transport = RequestsHTTPTransport(
    url="https://hub.snapshot.org/graphql",
    verify=True,
    retries=3,
    timeout = 600,
    retry_status_forcelist = [429, 500, 502, 503, 504, 524]
)
client = Client(transport=transport, fetch_schema_from_transport=True,execute_timeout=600)

def fetch_proposals(space_name):
    query = gql(f"""
    {{
    proposals(
        first: 1000, 
        skip: 0, 
        where: {{ 
            space: "{space_name}"
        }}, 
        orderBy: "created", 
        orderDirection: desc
    ) {{
        id
        title
        start
        end
        state
        choices
        scores
        scores_total
        scores_updated
        author
        type
        }}
    }}
    """)
    #running the query
    response = client.execute(query)
    proposals = response['proposals']
    # adding percentage scores to proposals
    for i in range(len(proposals)):
        if proposals[i]['scores_total'] != 0:
            proposals[i]['percentage_scores'] = [proposals[i]['scores'][j]/sum(proposals[i]['scores']) for j in range(len(proposals[i]['scores']))]
        else:
            proposals[i]['percentage_scores'] = [0 for _ in range(len(proposals[i]['scores']))]

    # converting to dataframe
    proposals_df = pd.DataFrame(proposals)
    # deleting rows with zero engagement
    proposals_df = proposals_df[proposals_df['scores_total'] != 0]

    return proposals_df

def fetch_votes(proposal_id,limit_to_50k=False):
    votes = []
    # first get the voting power of the top vote
    first_vote_query = gql(f"""
      {{
        votes(
          first: 1000,
          skip: 0,
          where: {{ 
              proposal: "{proposal_id}" 
          }},
          orderBy: "vp",
          orderDirection: desc,
          ) 
          {{
          voter
          vp
          choice
          created
          }}
      }}
      """)
    first_vote = client.execute(first_vote_query)
    if first_vote["votes"]:
      votes.extend(first_vote["votes"])
    else:
      return votes
    fetch_flag = True
    prev_vp = votes[-1]['vp']
    #flag for skipping first skip iter inside the for loop
    # since for the 1st instance we already fetch the initial 1k votes
    skip_first_iter = True
    # repeat till all votes are fetched
    while fetch_flag:
      #extract voting power
      curr_vp = prev_vp
      # stop if more than 50k votes collected (currently only executed for stargate)
      if limit_to_50k:
        if len(votes) >= 50000:
            break
      # get first 5000 votes less than this vp
      for skip in range(0,6000,1000):
        if skip_first_iter:
          skip_first_iter = False
          continue
        votes_query = gql(f"""
        {{
          votes(
            first: 1000,
            skip: {skip},
            where: {{ 
                proposal: "{proposal_id}",
                vp_lt: {curr_vp}
            }},
            orderBy: "vp",
            orderDirection: desc,
            ) 
            {{
            voter
            vp
            choice
            created
            }}
        }}
        """)
        votes_subset = client.execute(votes_query)
        if votes_subset["votes"]:
          votes.extend(votes_subset["votes"])
          if votes[-1]['vp'] == prev_vp:
            prev_vp = prev_vp - 1
          prev_vp = votes[-1]['vp']
        else:
          fetch_flag = False
          break

    return votes

def map_choice_value(row):
    raw_choice = row['choice']
    if isinstance(raw_choice, int):
        return raw_choice
    elif isinstance(raw_choice, dict):
        # For ranked or weighted votes: return top choice
        if raw_choice:
            top_choice = max(raw_choice, key=raw_choice.get)
            return top_choice
    elif isinstance(raw_choice, list):
        # For approval votes: return first approved choice
        if raw_choice:
            return raw_choice[0]
    elif isinstance(raw_choice, str):
        # The snapshot api returns str type for faulty votes
        return np.nan
    else:
        raise ValueError(f"Unsupported datatype {type(raw_choice)} returned by Snapshot")

# function to add new column containing voter participation and index for proposal engagement
def enhance_voter_proposal_matrix(df):
    if "voter_participation" in df.columns:
        df = df.drop('proposal_engagement')
        df = df.drop('voter_participation', axis=1)
    df['voter_participation'] = df.sum(axis=1)
    proposal_columns = [col for col in df.columns if col != 'voter_participation']
    proposal_engagement = df[proposal_columns].sum(axis=0)
    proposal_engagement.name = 'proposal_engagement'
    df = pd.concat([df, proposal_engagement.to_frame().T])
    return df

def sort_voters_by_participation(M):
    M['voter_participation'] = M.sum(axis=1)
    return M.sort_values(by='voter_participation', ascending=False)

def get_first_voter_with_empty_cell(M):
    for idx, row in M.iterrows():
        if (row == 0).any():
            return idx
    return None

def proposals_not_voted_on_by(M, voter):
    return M.columns[M.loc[voter] == 0].tolist()

def remove_proposals(M, proposals):
    return M.drop(columns=proposals)

def remove_voters_with_empty_cells(M):
    return M[(M != 0).all(axis=1)]

def num_proposals(M):
    return M.shape[1]

# function to build dense matrix from sparse matrix
def build_dataset(M, tau, k=1000):
    M = M.copy()
    # increment tau by one since we will also have an extra column for the voter participation
    tau += 1
    for _ in range(k):
        M = sort_voters_by_participation(M)  
        v = get_first_voter_with_empty_cell(M)
        if v is None:
            break

        P_prune = proposals_not_voted_on_by(M, v)
        if num_proposals(M) - len(P_prune) < tau:
            keep_count = num_proposals(M) - tau
            P_prune = P_prune[:keep_count]
        M = remove_proposals(M, P_prune)
        if num_proposals(M) <= tau:
            break
    M_D = remove_voters_with_empty_cells(M)
    return M_D

# function to vectorize votes for cosine similiarity
def encode_labels(df,choices,vectorized_labels_mapping):
    result = df.copy()
    vec_len = max([len(choices[prop]) for prop in choices.keys()])
    for col in df.columns:
        flag = True
        for choice_label in choices[col]:
            if choice_label not in vectorized_labels_mapping.keys():
                flag = False
                break
        for index in df.index:
            if flag:
                result.at[index,col] = vectorized_labels_mapping[choices[col][result.loc[index,col] - 1]]
            else:
                encoded_value = np.nan
                result.at[index,col] = encoded_value
    result = result.dropna(axis=1)
    return result

def calculate_cosine_similarity(voter_vector, delegate_matrix):
    cos_sim_all = []
    voter_votes = voter_vector.values.tolist()[0]
    for delegate in delegate_matrix.index:
        delegate_votes = delegate_matrix.loc[delegate].values.tolist()
        cos_sim_all.append(np.dot(voter_votes,delegate_votes)/(np.linalg.norm(voter_votes) * np.linalg.norm(delegate_votes)))
    return cos_sim_all

def calculate_eskin_similarity(voter_vector,delegate_matrix,freq_dist):
    eskin_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        eskin_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                eskin_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                eskin_sim_del.append(1)
            else:
                eskin_sim_del.append(len(freq_dist[prop].keys())**2/(len(freq_dist[prop].keys())**2 + 2))
        eskin_sim_all.append(sum(eskin_sim_del)/len(eskin_sim_del))
    return eskin_sim_all

def calculate_iof_similarity(voter_vector, delegate_matrix,freq_dist):
    iof_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        iof_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                iof_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                iof_sim_del.append(1)
            else:
                iof_sim_del.append(1/(1+ (np.log(freq_dist[prop][voter_vector[prop].item()]) * np.log(freq_dist[prop][delegate_matrix.loc[possible_delegate,prop]]))))
        iof_sim_all.append(sum(iof_sim_del)/len(iof_sim_del))
    return iof_sim_all

def calculate_of_similarity(voter_vector, delegate_matrix,freq_dist):
    of_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    N = len(delegate_matrix.index) + 1
    for possible_delegate in delegate_matrix.index:
        of_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                of_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                of_sim_del.append(1)
            else:
                of_sim_del.append(1/(1+ (np.log(N/freq_dist[prop][voter_vector[prop].item()]) * np.log(N/freq_dist[prop][delegate_matrix.loc[possible_delegate,prop]]))))
        of_sim_all.append(sum(of_sim_del)/len(of_sim_del))
    return of_sim_all

def calculate_lin_similarity(voter_vector, delegate_matrix,freq_dist):
    lin_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        lin_sim_del = []
        weight_part = 0
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                lin_sim_del.append(0)
                weight_part += np.log(freq_dist[prop][voter_vector[prop].item()]/sum(freq_dist[prop].values()))
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                lin_sim_del.append(2*np.log(freq_dist[prop][voter_vector[prop].item()]/sum(freq_dist[prop].values())))
                weight_part += np.log(freq_dist[prop][voter_vector[prop].item()]/sum(freq_dist[prop].values())) + np.log(freq_dist[prop][delegate_matrix.loc[possible_delegate,prop]]/sum(freq_dist[prop].values()))
            else:
                lin_sim_del.append(2*np.log((freq_dist[prop][voter_vector[prop].item()]/sum(freq_dist[prop].values())) + (freq_dist[prop][delegate_matrix.loc[possible_delegate,prop]]/sum(freq_dist[prop].values()))))
                weight_part += np.log(freq_dist[prop][voter_vector[prop].item()]/sum(freq_dist[prop].values())) + np.log(freq_dist[prop][delegate_matrix.loc[possible_delegate,prop]]/sum(freq_dist[prop].values()))

        lin_sim_all.append(sum(lin_sim_del)/weight_part)
    return lin_sim_all

def calculate_goodall1_similarity(voter_vector, delegate_matrix,freq_dist):
    goodall1_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        goodall1_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                goodall1_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                sum_part = 0
                for q in freq_dist[prop].keys():
                    if freq_dist[prop][q] <= freq_dist[prop][voter_vector[prop].item()]:
                        sum_part +=  (freq_dist[prop][q])*(freq_dist[prop][q]-1)
                sum_part = sum_part/(sum(freq_dist[prop].values())*(sum(freq_dist[prop].values())-1))
                goodall1_sim_del.append(1-sum_part)
            else:
                goodall1_sim_del.append(0)

        goodall1_sim_all.append(sum(goodall1_sim_del)/len(goodall1_sim_del))
    return goodall1_sim_all

def calculate_goodall2_similarity(voter_vector, delegate_matrix,freq_dist):
    goodall2_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        goodall2_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                goodall2_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                sum_part = 0
                for q in freq_dist[prop].keys():
                    if freq_dist[prop][q] >= freq_dist[prop][voter_vector[prop].item()]:
                        sum_part +=  (freq_dist[prop][q])*(freq_dist[prop][q]-1)
                sum_part = sum_part/(sum(freq_dist[prop].values())*(sum(freq_dist[prop].values())-1))
                goodall2_sim_del.append(1-sum_part)
            else:
                goodall2_sim_del.append(0)

        goodall2_sim_all.append(sum(goodall2_sim_del)/len(goodall2_sim_del))
    return goodall2_sim_all

def calculate_goodall3_similarity(voter_vector, delegate_matrix,freq_dist):
    goodall3_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        goodall3_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                goodall3_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                goodall3_sim_del.append(1-((freq_dist[prop][voter_vector[prop].item()]) * (freq_dist[prop][voter_vector[prop].item()] - 1)/(sum(freq_dist[prop].values())*(sum(freq_dist[prop].values()) - 1))))
            else:
                goodall3_sim_del.append(0)

        goodall3_sim_all.append(sum(goodall3_sim_del)/len(goodall3_sim_del))
    return goodall3_sim_all

def calculate_goodall4_similarity(voter_vector, delegate_matrix,freq_dist):
    goodall4_sim_all = []
    voter_vector = voter_vector.dropna(axis=1)
    for possible_delegate in delegate_matrix.index:
        goodall4_sim_del = []
        for prop in voter_vector.columns:
            if pd.isna(delegate_matrix.loc[possible_delegate,prop]):
                goodall4_sim_del.append(0)
            elif voter_vector[prop].item() == delegate_matrix.loc[possible_delegate,prop]:
                goodall4_sim_del.append((freq_dist[prop][voter_vector[prop].item()]) * (freq_dist[prop][voter_vector[prop].item()] - 1)/(sum(freq_dist[prop].values())*(sum(freq_dist[prop].values()) - 1)))
            else:
                goodall4_sim_del.append(0)

        goodall4_sim_all.append(sum(goodall4_sim_del)/len(goodall4_sim_del))
    return goodall4_sim_all

# calculating similiarity scores and return a df the k top delegates 
def recommend_delegates(voter_id, delegate_matrix, metric,freq_dist,top_k = 14):
    if metric == 'cosine_similarity':
        # Compute cosine similarity
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_cosine_similarity(voter_vector, delegate_matrix)

    elif metric == "overlap":
        #computing overlap similiarity
        voter_vector = delegate_matrix.loc[[voter_id]].values[0]
        delegate_matrix = delegate_matrix.drop(voter_id)
        delegate_matrix_list = delegate_matrix.values
        similarities = []
        for row in delegate_matrix_list:
            similarity = np.sum(voter_vector == row)/len(voter_vector)
            similarities.append(similarity)

    elif metric == "eskin":
        #computing similarity using eskin metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_eskin_similarity(voter_vector, delegate_matrix,freq_dist)
    
    elif metric == "iof":
        #computing similarity using iof metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_iof_similarity(voter_vector, delegate_matrix,freq_dist)
    
    elif metric == "of":
        #computing similarity using of metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_of_similarity(voter_vector, delegate_matrix,freq_dist)

    elif metric == "lin":
        #computing similarity using lin metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_lin_similarity(voter_vector, delegate_matrix,freq_dist)

    elif metric == "goodall1":
        #computing similarity using goodall1 metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_goodall1_similarity(voter_vector, delegate_matrix,freq_dist)

    elif metric == "goodall2":
        #computing similarity using goodall2 metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_goodall2_similarity(voter_vector, delegate_matrix,freq_dist)
    
    elif metric == "goodall3":
        #computing similarity using goodall2 metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_goodall3_similarity(voter_vector, delegate_matrix,freq_dist)

    elif metric == "goodall4":
        #computing similarity using goodall2 metric
        voter_vector = delegate_matrix.loc[[voter_id]]
        delegate_matrix = delegate_matrix.drop(voter_id)
        similarities = calculate_goodall4_similarity(voter_vector, delegate_matrix,freq_dist)

    # Create DataFrame of results
    sim_df = pd.DataFrame({
        'delegate': delegate_matrix.index,
        'similarity': similarities
    }).sort_values(by='similarity', ascending=False)
    return sim_df.head(top_k)

def evaulate_accuracy(voter_id,delegate_id,eval_set):
    predicted = eval_set.loc[delegate_id].values
    ground_truth = eval_set.loc[voter_id].values
    result = np.equal(predicted,ground_truth)
    return sum(result)/len(result)

def create_boxplots(data_dict, title="Comparing different Similiarity Metrics on Validation set", ylabel="Accuracy"):
    if not data_dict:
        raise ValueError("The dictionary is empty. Please provide at least one dataset.")
    labels = list(data_dict.keys())
    # changing name for easier visualization
    labels[0] = "cosine"
    data = list(data_dict.values())
    plt.figure(figsize=(12, 10))
    plt.boxplot(data, vert=True, patch_artist=True, labels=labels, showmeans=True, meanprops={"marker":"o","markerfacecolor":"red","markeredgecolor":"black","markersize":4})
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=90)
    plt.show()

def create_scatterplots(data_dict, title="Visualizing the recommendation accuracies", ylabel="Accuracy"):
    if not data_dict:
        raise ValueError("The dictionary is empty. Please provide at least one dataset.")
    plt.figure(figsize=(12, 8))
    for label, points in data_dict.items():
        if not points:
            continue
        x = range(len(points))
        y = points
        plt.scatter(x, y, label=label, s=60, alpha=0.7)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

def filter_common_proposals(delegate_i_votes,delegate_j_votes):
    mask = delegate_i_votes.notna() & delegate_j_votes.notna()
    common_proposals = list(delegate_i_votes.index[mask])
    return common_proposals

def get_freq_dist_vp_matrix(vp_matrix):
    freq_dict = {}
    for col in vp_matrix.columns:
        freq_dict[col] = vp_matrix[col].value_counts(dropna=True).to_dict()
    return freq_dict

def get_accuracy_for_metric(voter_proposal_matrix,recommended_delegates,metric,index):
    voter_id = recommended_delegates[metric][index][0]
    delegate_id = recommended_delegates[metric][index][1]
    
    # Check if both voter and delegate exist in the matrix
    if voter_id not in voter_proposal_matrix.index or delegate_id not in voter_proposal_matrix.index:
        return np.nan 
    common_proposals = filter_common_proposals(
        voter_proposal_matrix.loc[voter_id],
        voter_proposal_matrix.loc[delegate_id]
    )
    if not common_proposals:
        return np.nan
    
    same_votes = [prop for prop in common_proposals if 
                  voter_proposal_matrix.loc[voter_id][prop] == 
                  voter_proposal_matrix.loc[delegate_id][prop]]
    return len(same_votes)/len(common_proposals)

def get_precision(recommender_results,random_results):
    precision_list = []
    errors_below_baseline = []
    for result_index in range(len(random_results)):
        if recommender_results[result_index] >= random_results[result_index]:
            precision_list.append(1)
        else:
            precision_list.append(0)
            errors_below_baseline.append(random_results[result_index] - recommender_results[result_index])
    return precision_list,errors_below_baseline

def get_recall(recommender_results,perfect_delegate_existence):
    recall_list = []
    for result_index in range(len(recommender_results)):
        if perfect_delegate_existence[result_index]:
            if recommender_results[result_index] == 1:
                recall_list.append(1)
            else:
                recall_list.append(0)
    return recall_list

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_proposal_type_count(proposals_df, controversial_proposals=None):
    props_by_type = {}
    for row in proposals_df.itertuples():
        if controversial_proposals is not None:
            if row.id not in list(controversial_proposals['id']):
                continue
        if row.type in props_by_type.keys():
            props_by_type[row.type] += 1
        else:
            props_by_type[row.type] = 1
    return props_by_type

def get_proposal_choices(name):
    proposals_df = pd.read_json(f'data/snapshot_proposals_{name}.json')
    return proposals_df[['id', 'choices']].set_index('id').to_dict()['choices']

def evaluate_similarity_metrics(voting_df, recommendation_dataset, recommended_delegates, cfg, visualize_results):
    #  Filter voting data
    voting_df = voting_df[voting_df.voter.isin(recommendation_dataset.index)]
    
    # Create voter proposal matrix
    voter_proposal_matrix = voting_df.pivot_table(
        index='voter',
        columns='proposal_id',
        values='choice_value',
        aggfunc='first',
        fill_value=np.nan
    )
    
    # Initialize results dict
    eval_result = {
        'cosine_similarity': [], 'overlap': [], 'eskin': [], 'iof': [], 
        'of': [], 'lin': [], 'goodall1': [], 'goodall2': [], 'goodall3': [], 
        'goodall4': [], "baseline": []
    }

    # Get number of evaluation points
    eval_points = len(recommended_delegates['cosine_similarity'])

    # Evaluate each metric
    for i in range(eval_points):
        # Evaluate all similarity metrics
        for metric in eval_result.keys():
            if metric == "baseline":
                continue
            eval_result[metric].append(get_accuracy_for_metric(voter_proposal_matrix, recommended_delegates, metric, i))
        
        # evaluating random delegation (baseline)
        baseline_for_voter = []
        voter_id = recommended_delegates['overlap'][i][0]
        # Check if voter exists in matrix before proceeding
        if voter_id in voter_proposal_matrix.index:
            for possible_delegate in voter_proposal_matrix.drop(voter_id).index:
                common_proposals = filter_common_proposals(
                    voter_proposal_matrix.loc[voter_id], 
                    voter_proposal_matrix.loc[possible_delegate]
                )
                if common_proposals:
                    same_votes = [prop for prop in common_proposals if 
                                voter_proposal_matrix.loc[voter_id][prop] == 
                                voter_proposal_matrix.loc[possible_delegate][prop]]
                    baseline_for_voter.append(len(same_votes)/len(common_proposals))
            
            if baseline_for_voter:
                eval_result['baseline'].append(sum(baseline_for_voter)/len(baseline_for_voter))
            else:
                eval_result['baseline'].append(np.nan)
        else:
            eval_result['baseline'].append(np.nan)
        
    
    # Create accuracy statistics table
    accuracy_stats = {}
    for metric in eval_result.keys():
        metric_name = metric.replace('_', ' ').title()
        accuracy_stats[metric_name] = pd.Series(eval_result[metric]).describe()
    
    accuracy_table = pd.DataFrame(accuracy_stats).T
    print("Accuracy")
    print(accuracy_table.round(4).to_string())
    print()
    
    if visualize_results:
        # Create boxplot
        create_boxplots(eval_result, f"Comparing accuracy of different similarity measures for {cfg['name']}")
    
    # Calculate outperformance ratio and underperformance error
    performance_metrics = {}
    for metric in eval_result.keys():
        if metric == "baseline":
            continue
        precision_list, errors_below_baseline = get_precision(eval_result[metric], eval_result["baseline"])
        performance_metrics[metric] = {
            'outperformance_ratio': sum(precision_list)/len(precision_list),
            'underperformance_error': sum(errors_below_baseline)/len(errors_below_baseline) if len(errors_below_baseline) != 0 else 0
        }
    performance_table = pd.DataFrame(performance_metrics).T
    print("Outperformance Ratio and Underperformance Error")
    print(performance_table.round(4).to_string())
    print()

    # Get perfect delegate existence
    perfect_delegate_existence = []
    for i in range(eval_points):
        delegate_found = False
        voter_id = recommended_delegates['overlap'][i][0]
        # Check if voter exists in matrix before proceeding
        if voter_id in voter_proposal_matrix.index:
            for possible_delegate in voter_proposal_matrix.drop(voter_id).index:
                common_proposals = filter_common_proposals(
                    voter_proposal_matrix.loc[voter_id], 
                    voter_proposal_matrix.loc[possible_delegate]
                )
                if common_proposals:
                    same_votes = [prop for prop in common_proposals if 
                                voter_proposal_matrix.loc[voter_id][prop] == 
                                voter_proposal_matrix.loc[possible_delegate][prop]]
                    if len(same_votes) == len(common_proposals):
                        delegate_found = True
                        break    
        perfect_delegate_existence.append(delegate_found)

    # Calculate perfect delegate recall
    recall_metrics = {}
    for metric in eval_result.keys():
        if metric == "baseline":
            continue
        recall_list = get_recall(eval_result[metric], perfect_delegate_existence)
        if recall_list:
            recall_metrics[metric] = sum(recall_list)/len(recall_list)
        else:
            recall_metrics[metric] = 'N/A'  # No perfect delegate found
    recall_table = pd.DataFrame.from_dict(recall_metrics, orient='index', columns=['perfect_delegate_recall'])
    print("Perfect Delegate Recall")
    print(recall_table.round(2).to_string())
    print()