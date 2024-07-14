import torch
import torch.nn as nn

from .SeqContext import SeqContext1
from .SeqContext import SeqContext2
from .GNN import GNN1
from .GNN import GNN2

from .Classifier import Classifier
from .functions import global_graphify
from .functions import local_graphify
import bigraph



log = bigraph.utils.get_logger()


class BiGraph(nn.Module):
    def __init__(self, args):
        super(BiGraph, self).__init__() 
        u_dim = 100
        if args.rnn == "transformer":
            g_dim = args.hidden_size
        else:
            g_dim = 200
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size*2
        
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
        }

        if args.dataset and args.emotion == "multilabel":
            dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]
        self.concat_gin_gout = args.concat_gin_gout 
        self.local_global_cat = args.local_global_cat
        self.wp = args.wp
        self.wf = args.wf
        self.n_neighbors = args.n_neighbors
        self.device = args.device
        self.alpha = args.alpha
        
        self.rnn1 = SeqContext1(u_dim, g_dim, args)
        self.rnn2 = SeqContext2(u_dim, g_dim, args)
        self.gcn1 = GNN1(g_dim, h1_dim, h2_dim, args)
        self.gcn2 = GNN2(g_dim, h1_dim, h2_dim, args)

        if args.concat_gin_gout:
            self.clf = Classifier(
                g_dim + h2_dim * args.gnn_nheads, hc_dim, tag_size, args
            )
        else:
            self.clf = Classifier(h2_dim * 2 * args.gnn_nheads, hc_dim, tag_size, args)
            '''
            if concatenate graph_global and graph_local, change line 73, BiGraph.py
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)
            ------->
            self.clf = Classifier(h2_dim * 2 * args.gnn_nheads, hc_dim, tag_size, args)
            '''
        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        # print(edge_type_to_idx)
        log.debug(self.edge_type_to_idx)
    
        
    def get_rep(self, data):

        local_global_cat = self.local_global_cat
        node_features1 = self.rnn1(data["text_len_tensor"], data["input_tensor"])
        node_features2 = self.rnn2(data["text_len_tensor"], data["input_tensor"])     

        if local_global_cat == 1: 
            features, edge_index, edge_type, edge_index_lengths = local_graphify(
                node_features1,
                data["text_len_tensor"],
                data["speaker_tensor"],
                self.wp,
                self.wf,
                self.edge_type_to_idx,
                self.device)
            graph_out = self.gcn1(features, edge_index, edge_type)
            
        elif local_global_cat == 2:
            global_node_features, global_edges, global_edge_type = global_graphify(data, 
                self.device,
                self.edge_type_to_idx,
                self.n_neighbors,
                node_features2)
            graph_out = self.gcn2(global_node_features, global_edges, global_edge_type)
            features = global_node_features
            
        elif local_global_cat == 0:
            features, edge_index, edge_type, edge_index_lengths = local_graphify(
                node_features1,
                data["text_len_tensor"],
                data["speaker_tensor"],
                self.wp,
                self.wf,
                self.edge_type_to_idx,
                self.device)
            global_node_features, global_edges, global_edge_type = global_graphify(data, 
                self.device,
                self.edge_type_to_idx,
                self.n_neighbors,
                node_features2)
            graph_local = self.gcn1(features, edge_index, edge_type)

            graph_global = self.gcn2(global_node_features, global_edges, global_edge_type)

            graph_out = torch.cat([graph_global,graph_local],dim=1)

            '''
            if concatenate graph_global and graph_local, change line 73, BiGraph.py
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)
            ------->
            self.clf = Classifier(h2_dim * 2 * args.gnn_nheads, hc_dim, tag_size, args)
            '''
        else: print("wrong local_global_cat parameter!")
        return graph_out, features

    def forward(self, data):
        
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            out = self.clf(
                torch.cat([features, graph_out], dim=-1), data["text_len_tensor"]
            )
        else:
            out = self.clf(graph_out, data["text_len_tensor"])

        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            loss = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )

        return loss
