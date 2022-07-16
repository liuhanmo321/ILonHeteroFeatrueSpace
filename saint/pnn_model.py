from .model import *

def freeze_model(model, freeze):
    for param in model.parameters():
        param.requires_grad = freeze
    return

class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        num_tasks = 3,
        dim_head = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        y_dim = 2,
        device = None,
        extractor_type = 'transformer'
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # create category embeddings table

        total_tokens = sum(categories)
        num_categorical = len(categories)
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.categories_offset = [categories_offset]
        # categories_offset = torch.unsqueeze(categories_offset, 0)
        print('categorical_offset', categories_offset)

        self.num_continuous = [num_continuous]

        if extractor_type != 'transformer' and extractor_type != 'mlp':
            depth = int(depth / 2)
        # extractor parameters
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.device = device
        self.num_tasks = num_tasks
        self.extractor_type = extractor_type
        # structure parameters

        self.hidden_dims = 128
        self.embed_dim = 32

        self.embeds = nn.ModuleList([nn.Embedding(total_tokens, self.dim)])
        self.simple_MLP = nn.ModuleList(
                [nn.ModuleList(
                    [simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)]
                )]
            )
        
        self.relu = nn.ReLU()

        if extractor_type == 'mlp':
            self.specific_unifier = nn.ModuleList([UnifierModel(num_categorical, num_continuous, dim, attn_dropout)])            
        
        if extractor_type == 'transformer':
            self.specific_extractor = nn.ModuleList(
                [nn.ModuleList(
                    [singleTransformer(
                        dim = dim,
                        heads = heads,
                        dim_head = dim_head,
                        attn_dropout = attn_dropout,
                        ff_dropout = ff_dropout
                    ) for _ in range(depth)]
                ) for _ in range(num_tasks)]
            )
        elif extractor_type == 'rnn':
            self.specific_extractor = nn.ModuleList(
                [nn.ModuleList(
                    [RNNModel(
                        dim = dim,
                        heads = heads,
                        depth = 1,
                        dim_head = dim_head,
                        attn_dropout = attn_dropout,
                        ff_dropout = ff_dropout
                    ) for _ in range(depth)]
                ) for _ in range(num_tasks)]
            )
        elif extractor_type == 'gru':
            self.specific_extractor = nn.ModuleList(
                [nn.ModuleList(
                    [GRUModel(
                        dim = dim,
                        heads = heads,
                        depth = 1,
                        dim_head = dim_head,
                        attn_dropout = attn_dropout,
                        ff_dropout = ff_dropout
                    ) for _ in range(depth)]
                ) for _ in range(num_tasks)]
            )        
        elif extractor_type == 'mlp':
            self.specific_extractor = nn.ModuleList(
                [nn.ModuleList(
                    [MLPModel(
                        dim = dim,
                        depth = int(self.depth / 4),
                        dropout=attn_dropout
                    ) for _ in range(depth)]
                ) for _ in range(num_tasks)]                
            )

        self.specific_classifier_ly1 = nn.ModuleList([nn.Linear(dim, int(self.hidden_dims / 2)) for _ in range(num_tasks)])
        self.specific_classifier_ly2 = nn.ModuleList([nn.Linear(int(self.hidden_dims / 2), y_dim) for _ in range(num_tasks)])

        self.layer_embeds = nn.ModuleList(
            [nn.ModuleList(
                [nn.Embedding(1,task) for _ in range(depth + 1)]
            ) for task in range(num_tasks) if task > 0]
        )

        if extractor_type == 'transformer':
            self.enc_connect_layer = nn.ModuleList([
                    nn.ModuleList(
                        [nn.Sequential(
                            nn.Linear(task * dim, dim),
                            nn.ReLU(),
                            singleTransformer(
                                dim = dim,
                                heads = heads,
                                dim_head = dim_head,
                                attn_dropout = attn_dropout,
                                ff_dropout = ff_dropout
                            )
                        ) for _ in range(depth - 1)]
                    ) 
                for task in range(num_tasks) if task > 0
                ])
        elif extractor_type == 'rnn':
            self.enc_connect_layer = nn.ModuleList([
                    nn.ModuleList(
                        [nn.Sequential(
                            nn.Linear(task * dim, dim),
                            nn.ReLU(),
                            RNNModel(
                                dim = dim,
                                heads = heads,
                                depth = 1,
                                dim_head = dim_head,
                                attn_dropout = attn_dropout,
                                ff_dropout = ff_dropout
                            )
                        ) for _ in range(depth - 1)]
                    ) 
                for task in range(num_tasks) if task > 0
                ])  
        elif extractor_type == 'gru':
            self.enc_connect_layer = nn.ModuleList([
                    nn.ModuleList(
                        [nn.Sequential(
                            nn.Linear(task * dim, dim),
                            nn.ReLU(),
                            GRUModel(
                                dim = dim,
                                heads = heads,
                                depth = 1,
                                dim_head = dim_head,
                                attn_dropout = attn_dropout,
                                ff_dropout = ff_dropout
                            )
                        ) for _ in range(depth - 1)]
                    ) 
                for task in range(num_tasks) if task > 0
                ])  
        elif extractor_type == 'mlp':
            self.enc_connect_layer = nn.ModuleList([
                    nn.ModuleList(
                        [nn.Sequential(
                            nn.Linear(task * dim, dim),
                            nn.ReLU(),
                            MLPModel(
                                dim = dim,
                                depth = 1,
                                dropout = attn_dropout,
                            )
                        ) for _ in range(depth - 1)]
                    ) 
                for task in range(num_tasks) if task > 0
                ])                            


        self.fc_connect_layer = nn.ModuleList(
            [
                simple_MLP([task * dim, dim, int(self.hidden_dims / 2)]) for task in range(num_tasks) if task > 0
            ]
            )
        
        self.pred_connect_layer = nn.ModuleList(
            [
                # nn.Linear(int(self.hidden_dims / 2), y_dim) for task in range(num_tasks) if task > 0/
                simple_MLP([task * int(self.hidden_dims / 2), int(self.hidden_dims / 2), y_dim]) for task in range(num_tasks) if task > 0
            ]
        )

        # end modification        
        
    def forward(self, x_categ, x_cont, data_id):
        for d in range(self.depth):
            if d == 0:
                if self.extractor_type == 'mlp':
                    unified_output = self.specific_unifier[data_id](x_categ, x_cont)
                    h = self.specific_extractor[data_id][d](unified_output)
                    if data_id > 0:
                        h_prev = [self.specific_extractor[temp_id][d](unified_output) for temp_id in range(data_id)]
                else:
                    h = self.specific_extractor[data_id][d](x_categ, x_cont)
                    if data_id > 0:
                        h_prev = [self.specific_extractor[temp_id][d](x_categ, x_cont) for temp_id in range(data_id)]
            else:
                h_pre = self.specific_extractor[data_id][d](h)
                if data_id > 0:
                    cat_dim = 1 if self.extractor_type == 'mlp' else 2
                    h_pre = h_pre + self.enc_connect_layer[data_id - 1][d-1](torch.cat([self.layer_embeds[data_id-1][d-1].weight[0][temp_id] * h_prev[temp_id] for temp_id in range(data_id)], cat_dim))
                    h_prev = [self.specific_extractor[temp_id][d](h_prev[temp_id]) for temp_id in range(data_id)]                
                    
                h = h_pre
        
        feat_idx = 0 if self.extractor_type == 'transformer' else -1
        if self.extractor_type != 'mlp':
            h = h[:, feat_idx, :]        
            if data_id > 0:
                h_prev = [prev[:, feat_idx, :] for prev in h_prev]    

        h_pre = self.specific_classifier_ly1[data_id](h)
        if data_id > 0:
            h_pre = h_pre + self.fc_connect_layer[data_id - 1](torch.cat([self.layer_embeds[data_id-1][self.depth-1].weight[0][temp_id] * h_prev[temp_id] for temp_id in range(data_id)],1))
            h_prev = [self.specific_classifier_ly1[temp_id](h_prev[temp_id]) for temp_id in range(data_id)]
            
        h = self.relu(h_pre)
        
        h_pre = self.specific_classifier_ly2[data_id](h)
        if data_id > 0:
            h = h_pre + self.pred_connect_layer[data_id - 1](torch.cat([self.layer_embeds[data_id-1][self.depth].weight[0][temp_id] * h_prev[temp_id] for temp_id in range(data_id)],1))
        else:
            h = h_pre
        return h

    def add_task(self, categories, num_continuous, y_dim):
        total_tokens = sum(categories)
        self.num_continuous.append(num_continuous)
        num_categorical = len(categories)

        temp_categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        temp_categories_offset = temp_categories_offset.cumsum(dim = -1)[:-1]
        self.categories_offset.append(temp_categories_offset)

        print('categorical_offset', self.categories_offset)

        self.embeds.append(nn.Embedding(total_tokens, self.dim))
        self.simple_MLP.append(
            nn.ModuleList([simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)])
        )
        
        if self.extractor_type == 'mlp':
            self.specific_unifier.append(UnifierModel(num_categorical, num_continuous, self.dim, self.attn_dropout))
    
    def unfreeze_column(self, data_id):
        for temp_id in range(data_id):
            freeze = (temp_id == data_id)
            freeze_model(self.embeds[temp_id], freeze)
            freeze_model(self.simple_MLP[temp_id], freeze)

        for temp_id in range(self.num_tasks):
            freeze = (temp_id == data_id)
            freeze_model(self.specific_extractor[temp_id], freeze)
            freeze_model(self.specific_classifier_ly1[temp_id], freeze)
            freeze_model(self.specific_classifier_ly2[temp_id], freeze)
            if temp_id > 0:
                freeze_model(self.layer_embeds[temp_id - 1], freeze)
                freeze_model(self.fc_connect_layer[temp_id - 1], freeze)
                freeze_model(self.pred_connect_layer[temp_id - 1], freeze)
                freeze_model(self.enc_connect_layer[temp_id - 1], freeze)
        
        for temp_id in range(data_id + 1):
            if self.extractor_type == 'mlp':
                freeze_model(self.specific_unifier[temp_id], freeze)
