import os
import math
import pickle
import numpy as np
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from torch_geometric.seed import seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import CitationFull, Coauthor, Flickr, RelLinkPredDataset, WordNet18, WordNet18RR
from torch_geometric.utils import train_test_split_edges, k_hop_subgraph, negative_sampling, to_undirected, is_undirected, to_networkx
from ogb.linkproppred import PygLinkPropPredDataset
from framework.utils import *


data_dir = './data'
external_drug_data_dir = '/root/autodl-tmp/data'
df_size = [i / 100 for i in range(10)] + [i / 10 for i in range(10)] + [i for i in range(10)]       # Df_size in percentage
seeds = [42, 21, 13, 87, 100]
graph_datasets = ['Cora', 'PubMed', 'DBLP', 'CS', 'ogbl-citation2', 'ogbl-collab'][4:]
kg_datasets = ['FB15k-237', 'WordNet18', 'WordNet18RR', 'ogbl-biokg'][-1:]
os.makedirs(data_dir, exist_ok=True)


num_edge_type_mapping = {
    'FB15k-237': 237,
    'WordNet18': 18,
    'WordNet18RR': 11
}

def train_test_split_edges_no_neg_adj_mask(data, val_ratio: float = 0.05, test_ratio: float = 0.1, two_hop_degree=None, kg=False):
    '''Avoid adding neg_adj_mask'''

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edge_attr = data.edge_attr
    if kg:
        edge_type = data.edge_type
    data.edge_index = data.edge_attr = data.edge_weight = data.edge_year = data.edge_type = None

    if not kg:
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        if edge_attr is not None:
            edge_attr = edge_attr[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    if two_hop_degree is not None:          # Use low degree edges for test sets
        low_degree_mask = two_hop_degree < 50

        low = low_degree_mask.nonzero().squeeze()
        high = (~low_degree_mask).nonzero().squeeze()

        low = low[torch.randperm(low.size(0))]
        high = high[torch.randperm(high.size(0))]

        perm = torch.cat([low, high])

    else:
        perm = torch.randperm(row.size(0))

    row = row[perm]
    col = col[perm]

    # Train
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    
    if kg:

        # data.edge_index and data.edge_type has reverse edges and edge types for message passing
        pos_edge_index = torch.stack([r, c], dim=0)
        # rev_pos_edge_index = torch.stack([r, c], dim=0)
        train_edge_type = edge_type[n_v + n_t:]
        # train_rev_edge_type = edge_type[n_v + n_t:] + edge_type.unique().shape[0]

        # data.edge_index = torch.cat((torch.stack([r, c], dim=0), torch.stack([r, c], dim=0)), dim=1)
        # data.edge_type = torch.cat([train_edge_type, train_rev_edge_type], dim=0)

        data.edge_index = pos_edge_index
        data.edge_type = train_edge_type
        
        # data.train_pos_edge_index and data.train_edge_type only has one direction edges and edge types for decoding
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_edge_type = train_edge_type
    
    else:
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        if edge_attr is not None:
            # out = to_undirected(data.train_pos_edge_index, edge_attr[n_v + n_t:])
            data.train_pos_edge_index, data.train_pos_edge_attr = out
        else:
            data.train_pos_edge_index = data.train_pos_edge_index
            # data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)
        
        assert not is_undirected(data.train_pos_edge_index)

    
    # Test
    r, c = row[:n_t], col[:n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    if kg:
        data.test_edge_type = edge_type[:n_t]
        neg_edge_index = negative_sampling_kg(
            edge_index=data.test_pos_edge_index,
            edge_type=data.test_edge_type)
    else:
        neg_edge_index = negative_sampling(
            edge_index=data.test_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.test_pos_edge_index.shape[1])

    data.test_neg_edge_index = neg_edge_index

    # Valid
    r, c = row[n_t:n_t+n_v], col[n_t:n_t+n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    if kg:
        data.val_edge_type = edge_type[n_t:n_t+n_v]
        neg_edge_index = negative_sampling_kg(
            edge_index=data.val_pos_edge_index,
            edge_type=data.val_edge_type)
    else:
        neg_edge_index = negative_sampling(
            edge_index=data.val_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.val_pos_edge_index.shape[1])

    data.val_neg_edge_index = neg_edge_index

    return data

def process_graph():
    for d in graph_datasets:

        if d in ['Cora', 'PubMed', 'DBLP']:
            dataset = CitationFull(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
        elif d in ['CS', 'Physics']:
            dataset = Coauthor(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
        elif d in ['Flickr']:
            dataset = Flickr(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
        elif 'ogbl' in d:
            dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)
        else:
            raise NotImplementedError

        print('Processing:', d)
        print(dataset)
        data = dataset[0]
        data.train_mask = data.val_mask = data.test_mask = None
        graph = to_networkx(data)

        # Get two hop degree for all nodes
        node_to_neighbors = {}
        for n in tqdm(graph.nodes(), desc='Two hop neighbors'):
            neighbor_1 = set(graph.neighbors(n))
            neighbor_2 = sum([list(graph.neighbors(i)) for i in neighbor_1], [])
            neighbor_2 = set(neighbor_2)
            neighbor = neighbor_1 | neighbor_2
            
            node_to_neighbors[n] = neighbor

        two_hop_degree = []
        row, col = data.edge_index
        mask = row < col
        row, col = row[mask], col[mask]
        for r, c in tqdm(zip(row, col), total=len(row)):
            neighbor_row = node_to_neighbors[r.item()]
            neighbor_col = node_to_neighbors[c.item()]
            neighbor = neighbor_row | neighbor_col
            
            num = len(neighbor)
            
            two_hop_degree.append(num)

        two_hop_degree = torch.tensor(two_hop_degree)

        for s in seeds:
            seed_everything(s)

            # D
            data = dataset[0]
            if 'ogbl' in d:
                data = train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.05, two_hop_degree=two_hop_degree)
            else:
                data = train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.05)
            print(s, data)

            with open(os.path.join(data_dir, d, f'd_{s}.pkl'), 'wb') as f:
                pickle.dump((dataset, data), f)

            # Two ways to sample Df from the training set
            ## 1. Df is within 2 hop local enclosing subgraph of Dtest
            ## 2. Df is outside of 2 hop local enclosing subgraph of Dtest
            
            # All the candidate edges (train edges)
            # graph = to_networkx(Data(edge_index=data.train_pos_edge_index, x=data.x))

            # Get the 2 hop local enclosing subgraph for all test edges
            _, local_edges, _, mask = k_hop_subgraph(
                data.test_pos_edge_index.flatten().unique(), 
                2, 
                data.train_pos_edge_index, 
                num_nodes=dataset[0].num_nodes)
            distant_edges = data.train_pos_edge_index[:, ~mask]
            print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

            in_mask = mask
            out_mask = ~mask

            # df_in_mask = torch.zeros_like(mask)
            # df_out_mask = torch.zeros_like(mask)

            # df_in_all_idx = in_mask.nonzero().squeeze()
            # df_out_all_idx = out_mask.nonzero().squeeze()
            # df_in_selected_idx = df_in_all_idx[torch.randperm(df_in_all_idx.shape[0])[:df_size]]
            # df_out_selected_idx = df_out_all_idx[torch.randperm(df_out_all_idx.shape[0])[:df_size]]
            
            # df_in_mask[df_in_selected_idx] = True
            # df_out_mask[df_out_selected_idx] = True

            # assert (in_mask & out_mask).sum() == 0
            # assert (df_in_mask & df_out_mask).sum() == 0


            # local_edges = set()
            # for i in range(data.test_pos_edge_index.shape[1]):
            #     edge = data.test_pos_edge_index[:, i].tolist()
            #     subgraph = get_enclosing_subgraph(graph, edge)
            #     local_edges = local_edges | set(subgraph[2])
            
            # distant_edges = graph.edges() - local_edges

            # print('aaaaaaa', len(local_edges), len(distant_edges))
            # local_edges = torch.tensor(sorted(list([i for i in local_edges if i[0] < i[1]])))
            # distant_edges = torch.tensor(sorted(list([i for i in distant_edges if i[0] < i[1]])))


            # df_in = torch.randperm(local_edges.shape[1])[:df_size]
            # df_out = torch.randperm(distant_edges.shape[1])[:df_size]

            # df_in = local_edges[:, df_in]
            # df_out = distant_edges[:, df_out]

            # df_in_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)
            # df_out_mask = torch.zeros(data.train_pos_edge_index.shape[1], dtype=torch.bool)

            # for row in df_in:
            #     i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
            #     df_in_mask[i] = True

            # for row in df_out:
            #     i = (data.train_pos_edge_index.T == row).all(axis=1).nonzero()
            #     df_out_mask[i] = True

            torch.save(
                {'out': out_mask, 'in': in_mask},
                os.path.join(data_dir, d, f'df_{s}.pt')
            )

def process_kg():
    for d in kg_datasets:

        # Create the dataset to calculate node degrees
        if d in ['FB15k-237']:
            dataset = RelLinkPredDataset(os.path.join(data_dir, d), d, transform=T.NormalizeFeatures())
            data = dataset[0]
            data.x = torch.arange(data.num_nodes)
            edge_index = torch.cat([data.train_edge_index, data.valid_edge_index, data.test_edge_index], dim=1)
            edge_type = torch.cat([data.train_edge_type, data.valid_edge_type, data.test_edge_type])
            data = Data(edge_index=edge_index, edge_type=edge_type)

        elif d in ['WordNet18RR']:
            dataset = WordNet18RR(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
            data = dataset[0]
            data.x = torch.arange(data.num_nodes)
            data.train_mask = data.val_mask = data.test_mask = None

        elif d in ['WordNet18']:
            dataset = WordNet18(os.path.join(data_dir, d), transform=T.NormalizeFeatures())
            data = dataset[0]
            data.x = torch.arange(data.num_nodes)

            # Use original split
            data.train_pos_edge_index = data.edge_index[:, data.train_mask]
            data.train_edge_type = data.edge_type[data.train_mask]

            data.val_pos_edge_index = data.edge_index[:, data.val_mask]
            data.val_edge_type = data.edge_type[data.val_mask]
            data.val_neg_edge_index = negative_sampling_kg(data.val_pos_edge_index, data.val_edge_type)

            data.test_pos_edge_index = data.edge_index[:, data.test_mask]
            data.test_edge_type = data.edge_type[data.test_mask]
            data.test_neg_edge_index = negative_sampling_kg(data.test_pos_edge_index, data.test_edge_type)

        elif 'ogbl' in d:
            dataset = PygLinkPropPredDataset(root=os.path.join(data_dir, d), name=d)

            split_edge = dataset.get_edge_split()
            train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
            entity_dict = dict()
            cur_idx = 0
            for key in dataset[0]['num_nodes_dict']:
                entity_dict[key] = (cur_idx, cur_idx + dataset[0]['num_nodes_dict'][key])
                cur_idx += dataset[0]['num_nodes_dict'][key]
            nentity = sum(dataset[0]['num_nodes_dict'].values())

            valid_head_neg = valid_edge.pop('head_neg')
            valid_tail_neg = valid_edge.pop('tail_neg')
            test_head_neg = test_edge.pop('head_neg')
            test_tail_neg = test_edge.pop('tail_neg')

            train = pd.DataFrame(train_edge)
            valid = pd.DataFrame(valid_edge)
            test = pd.DataFrame(test_edge)

            # Convert to global index
            train['head'] = [idx + entity_dict[tp][0] for idx, tp in zip(train['head'], train['head_type'])]
            train['tail'] = [idx + entity_dict[tp][0] for idx, tp in zip(train['tail'], train['tail_type'])]

            valid['head'] = [idx + entity_dict[tp][0] for idx, tp in zip(valid['head'], valid['head_type'])]
            valid['tail'] = [idx + entity_dict[tp][0] for idx, tp in zip(valid['tail'], valid['tail_type'])]

            test['head'] = [idx + entity_dict[tp][0] for idx, tp in zip(test['head'], test['head_type'])]
            test['tail'] = [idx + entity_dict[tp][0] for idx, tp in zip(test['tail'], test['tail_type'])]

            valid_pos_edge_index = torch.tensor([valid['head'], valid['tail']])
            valid_edge_type = torch.tensor(valid.relation)
            valid_neg_edge_index = torch.stack([valid_pos_edge_index[0], valid_tail_neg[:, 0]])

            test_pos_edge_index = torch.tensor([test['head'], test['tail']])
            test_edge_type = torch.tensor(test.relation)
            test_neg_edge_index = torch.stack([test_pos_edge_index[0], test_tail_neg[:, 0]])

            train_directed = train[train.head_type != train.tail_type]
            train_undirected = train[train.head_type == train.tail_type]
            train_undirected_uni = train_undirected[train_undirected['head'] < train_undirected['tail']]
            train_uni = pd.concat([train_directed, train_undirected_uni], ignore_index=True)

            train_pos_edge_index = torch.tensor([train_uni['head'], train_uni['tail']])
            train_edge_type = torch.tensor(train_uni.relation)

            r, c = train_pos_edge_index
            rev_edge_index = torch.stack([c, r])
            rev_edge_type = train_edge_type + 51

            edge_index = torch.cat([train_pos_edge_index, rev_edge_index], dim=1)
            edge_type = torch.cat([train_edge_type, rev_edge_type], dim=0)

            data = Data(
                x=torch.arange(nentity), edge_index=edge_index, edge_type=edge_type,
                train_pos_edge_index=train_pos_edge_index, train_edge_type=train_edge_type, 
                val_pos_edge_index=valid_pos_edge_index, val_edge_type=valid_edge_type, val_neg_edge_index=valid_neg_edge_index,
                test_pos_edge_index=test_pos_edge_index, test_edge_type=test_edge_type, test_neg_edge_index=test_neg_edge_index)

        else:
            raise NotImplementedError
            
        print('Processing:', d)
        print(dataset)
        
        for s in seeds:
            seed_everything(s)

            # D
            # data = train_test_split_edges_no_neg_adj_mask(data, test_ratio=0.05, two_hop_degree=two_hop_degree, kg=True)
            print(s, data)

            with open(os.path.join(data_dir, d, f'd_{s}.pkl'), 'wb') as f:
                pickle.dump((dataset, data), f)

            # Two ways to sample Df from the training set
            ## 1. Df is within 2 hop local enclosing subgraph of Dtest
            ## 2. Df is outside of 2 hop local enclosing subgraph of Dtest
            
            # All the candidate edges (train edges)
            # graph = to_networkx(Data(edge_index=data.train_pos_edge_index, x=data.x))

            # Get the 2 hop local enclosing subgraph for all test edges
            _, local_edges, _, mask = k_hop_subgraph(
                data.test_pos_edge_index.flatten().unique(), 
                2, 
                data.train_pos_edge_index, 
                num_nodes=dataset[0].num_nodes)
            distant_edges = data.train_pos_edge_index[:, ~mask]
            print('Number of edges. Local: ', local_edges.shape[1], 'Distant:', distant_edges.shape[1])

            in_mask = mask
            out_mask = ~mask

            torch.save(
                {'out': out_mask, 'in': in_mask},
                os.path.join(data_dir, d, f'df_{s}.pt')
            )


def process_drug(ccs_threshold: float = 0.6, pps_threshold: float = 0.5, val_ratio: float = 0.05, test_ratio: float = 0.1):
    os.makedirs(os.path.join(data_dir, 'drug'), exist_ok=True)

    # Load matrices
    cpi = np.load(os.path.join(external_drug_data_dir, 'CPI.npy'))  # [C, P]
    ccs = np.load(os.path.join(external_drug_data_dir, 'CCS.npy'))  # [C, C]
    pps = np.load(os.path.join(external_drug_data_dir, 'PPS.npy'))  # [P, P]
    comp_se = np.load(os.path.join(external_drug_data_dir, 'compound_se.npy'))  # [C, S]

    num_compound, num_protein = cpi.shape
    _, num_side_effect = comp_se.shape

    # Build CPI edges (compound -> protein), relation id 0
    cpi_pos = np.argwhere(cpi > 0)
    cpi_src = torch.tensor(cpi_pos[:, 0], dtype=torch.long)
    cpi_dst = torch.tensor(num_compound + cpi_pos[:, 1], dtype=torch.long)
    cpi_edge_index = torch.stack([cpi_src, cpi_dst], dim=0)
    cpi_edge_type = torch.zeros(cpi_edge_index.shape[1], dtype=torch.long)

    # Build CCS edges (compound <-> compound), relation id 1
    ccs_mask = (ccs >= ccs_threshold)
    ccs_triu = np.triu(ccs_mask, k=1)
    ccs_pos = np.argwhere(ccs_triu)
    if ccs_pos.size == 0:
        ccs_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        ccs_edge_index = torch.tensor(ccs_pos.T, dtype=torch.long)
    ccs_edge_type = torch.full((ccs_edge_index.shape[1],), 1, dtype=torch.long)

    # Build PPS edges (protein <-> protein), relation id 2
    pps_mask = (pps >= pps_threshold)
    pps_triu = np.triu(pps_mask, k=1)
    pps_pos = np.argwhere(pps_triu)
    if pps_pos.size == 0:
        pps_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        pps_src = torch.tensor(num_compound + pps_pos[:, 0], dtype=torch.long)
        pps_dst = torch.tensor(num_compound + pps_pos[:, 1], dtype=torch.long)
        pps_edge_index = torch.stack([pps_src, pps_dst], dim=0)
    pps_edge_type = torch.full((pps_edge_index.shape[1],), 2, dtype=torch.long)

    # Build compound-side effect edges (compound -> side_effect), relation id 3
    se_pos = np.argwhere(comp_se > 0)
    if se_pos.size == 0:
        se_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        se_src = torch.tensor(se_pos[:, 0], dtype=torch.long)
        se_dst = torch.tensor(num_compound + num_protein + se_pos[:, 1], dtype=torch.long)
        se_edge_index = torch.stack([se_src, se_dst], dim=0)
    se_edge_type = torch.full((se_edge_index.shape[1],), 3, dtype=torch.long)

    # Prepare CPI splits
    N = cpi_edge_index.shape[1]
    # Build global CPI positive set for filtered negative sampling
    all_cpi_pairs = set((int(h), int(t)) for h, t in cpi_edge_index.t().tolist())
    for s in seeds:
        seed_everything(s)
        perm = torch.randperm(N)
        n_v = int(math.floor(val_ratio * N))
        n_t = int(math.floor(test_ratio * N))

        # Train
        train_idx = perm[n_v + n_t:]
        train_cpi_edge_index = cpi_edge_index[:, train_idx]
        train_cpi_edge_type = cpi_edge_type[train_idx]

        # Val
        val_idx = perm[n_t:n_t + n_v]
        val_cpi_edge_index = cpi_edge_index[:, val_idx]
        val_cpi_edge_type = cpi_edge_type[val_idx]
        # Filtered negatives for CPI: corrupt head/tail with type constraints and avoid true edges
        val_neg_edge_index = []
        num_needed = val_cpi_edge_index.shape[1]
        while len(val_neg_edge_index) < num_needed:
            # batch sample
            batch = min(4096, num_needed - len(val_neg_edge_index))
            corrupt_head = torch.rand(batch) < 0.5
            # sample heads/tails
            heads = torch.randint(0, num_compound, (batch,))
            tails = torch.randint(0, num_protein, (batch,)) + num_compound
            # choose base pos edges to corrupt
            base_idx = torch.randint(0, val_cpi_edge_index.shape[1], (batch,))
            base_pos = val_cpi_edge_index[:, base_idx]
            new_heads = torch.where(corrupt_head, heads, base_pos[0])
            new_tails = torch.where(corrupt_head, base_pos[1], tails)
            # filter
            candidates = list(zip(new_heads.tolist(), new_tails.tolist()))
            for h, t in candidates:
                if (int(h), int(t)) not in all_cpi_pairs:
                    val_neg_edge_index.append((h, t))
                    if len(val_neg_edge_index) >= num_needed:
                        break
        val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).t().contiguous()

        # Test
        test_idx = perm[:n_t]
        test_cpi_edge_index = cpi_edge_index[:, test_idx]
        test_cpi_edge_type = cpi_edge_type[test_idx]
        # Filtered negatives for CPI
        test_neg_edge_index = []
        num_needed = test_cpi_edge_index.shape[1]
        while len(test_neg_edge_index) < num_needed:
            batch = min(4096, num_needed - len(test_neg_edge_index))
            corrupt_head = torch.rand(batch) < 0.5
            heads = torch.randint(0, num_compound, (batch,))
            tails = torch.randint(0, num_protein, (batch,)) + num_compound
            base_idx = torch.randint(0, test_cpi_edge_index.shape[1], (batch,))
            base_pos = test_cpi_edge_index[:, base_idx]
            new_heads = torch.where(corrupt_head, heads, base_pos[0])
            new_tails = torch.where(corrupt_head, base_pos[1], tails)
            candidates = list(zip(new_heads.tolist(), new_tails.tolist()))
            for h, t in candidates:
                if (int(h), int(t)) not in all_cpi_pairs:
                    test_neg_edge_index.append((h, t))
                    if len(test_neg_edge_index) >= num_needed:
                        break
        test_neg_edge_index = torch.tensor(test_neg_edge_index, dtype=torch.long).t().contiguous()

        # Message passing edges: include all train CPI edges and all similarity/SE edges
        mp_edge_index = [train_cpi_edge_index, ccs_edge_index, pps_edge_index, se_edge_index]
        mp_edge_index = torch.cat([ei for ei in mp_edge_index if ei.numel() > 0], dim=1) if any(ei.numel() > 0 for ei in mp_edge_index) else torch.empty((2, 0), dtype=torch.long)
        mp_edge_type = torch.cat([train_cpi_edge_type, ccs_edge_type, pps_edge_type, se_edge_type], dim=0) if mp_edge_index.numel() > 0 else torch.empty((0,), dtype=torch.long)

        # Add reverse edges for message passing
        if mp_edge_index.numel() > 0:
            r, c = mp_edge_index
            rev_edge_index = torch.stack([c, r], dim=0)
            rev_edge_type = mp_edge_type + 4  # num_edge_type = 4
            edge_index = torch.cat([mp_edge_index, rev_edge_index], dim=1)
            edge_type = torch.cat([mp_edge_type, rev_edge_type], dim=0)
        else:
            edge_index = mp_edge_index
            edge_type = mp_edge_type

        num_nodes = num_compound + num_protein + num_side_effect
        data = Data(
            x=torch.arange(num_nodes),
            edge_index=edge_index,
            edge_type=edge_type,
            train_pos_edge_index=mp_edge_index,  # not used by trainer for KG if prebuilt flag is set
            train_edge_type=mp_edge_type,
            val_pos_edge_index=val_cpi_edge_index,
            val_edge_type=val_cpi_edge_type,
            val_neg_edge_index=val_neg_edge_index,
            test_pos_edge_index=test_cpi_edge_index,
            test_edge_type=test_cpi_edge_type,
            test_neg_edge_index=test_neg_edge_index,
        )

        # Bookkeeping
        data.relation_to_id = {'CPI': 0, 'CCS': 1, 'PPS': 2, 'SE': 3}
        data.prebuilt = True
        data.dr_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
        data.num_compound = num_compound
        data.num_protein = num_protein
        data.num_side_effect = num_side_effect
        data.all_cpi_pos_edge_index = cpi_edge_index

        out_path = os.path.join(data_dir, 'drug', f'd_{s}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(({'meta': 'drug'}, data), f)
        print(f'Drug dataset processed for seed {s}. Saved to {out_path}')

        # Build Df masks (in/out) w.r.t. 2-hop enclosing subgraph of CPI test nodes
        # We create masks over the training message-passing edges (mp_edge_index),
        # but only CPI edges are eligible for deletion.
        num_train_cpi = train_cpi_edge_index.shape[1]
        _, local_edges, _, mask_edges = k_hop_subgraph(
            test_cpi_edge_index.flatten().unique(),
            2,
            mp_edge_index,
            num_nodes=num_nodes
        )
        in_mask = mask_edges.clone()
        out_mask = ~mask_edges
        # Restrict masks to CPI portion only; others set to False
        if in_mask.shape[0] != mp_edge_index.shape[1]:
            in_mask = in_mask.view(-1)
        if out_mask.shape[0] != mp_edge_index.shape[1]:
            out_mask = out_mask.view(-1)
        # Zero out non-CPI regions
        non_cpi_mask = torch.ones(mp_edge_index.shape[1], dtype=torch.bool)
        non_cpi_mask[:num_train_cpi] = False
        in_mask[non_cpi_mask] = False
        out_mask[non_cpi_mask] = False

        torch.save({'in': in_mask, 'out': out_mask}, os.path.join(data_dir, 'drug', f'df_{s}.pt'))

def main():
    # process_graph()
    # process_kg()
    process_drug()

if __name__ == "__main__":
    main()
