from .model import *
import copy

celossmean = nn.CrossEntropyLoss(reduction='mean')

class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        y_dim = 2,
        extractor_type = 'transformer'
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # create category embeddings table

        total_tokens = sum(categories)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        self.categories_offset = [categories_offset]
        # categories_offset = torch.unsqueeze(categories_offset, 0)
        print('categorical_offset', categories_offset)

        self.unlabeled_categories_offset = []

        # self.num_continuous = num_continuous
        self.num_continuous = [num_continuous]
        self.unlabeled_num_continuous = []

        # extractor parameters
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
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
        
        # start modification

        if extractor_type == 'transformer':
            self.shared_extractor = Transformer(
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    dim_head = dim_head,
                    attn_dropout = attn_dropout,
                    ff_dropout = ff_dropout
                )
        elif extractor_type == 'rnn':
            self.shared_extractor = RNNModel(
                    dim = dim,
                    depth = int(depth / 2),
                    heads = heads,
                    dim_head = dim_head,
                    attn_dropout = attn_dropout,
                    ff_dropout = ff_dropout
                )            
        elif extractor_type == 'gru':
            self.shared_extractor = GRUModel(
                    dim = dim,
                    depth = int(self.depth / 2),
                    heads = heads,
                    dim_head = dim_head,
                    attn_dropout = attn_dropout,
                    ff_dropout = ff_dropout
                )  

        # self.task_classifier = nn.ModuleList([simple_MLP([dim, int(self.hidden_dims / 2), y_dim])])
        
        self.class_classifier = nn.ModuleList(
            [nn.ModuleList(
                [simple_MLP([dim, int(self.hidden_dims / 2), 1]) for _ in range(y_dim)]
            )]
        )

        # end modification        

    def add_task(self, categories, num_continuous, y_dim):
        total_tokens = sum(categories)
        self.num_continuous.append(num_continuous)

        temp_categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        temp_categories_offset = temp_categories_offset.cumsum(dim = -1)[:-1]
        self.categories_offset.append(temp_categories_offset)

        print('categorical_offset', self.categories_offset)

        self.embeds.append(nn.Embedding(total_tokens, self.dim))
        self.simple_MLP.append(
            nn.ModuleList([simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)])
        )
        
        # self.task_classifier.append(simple_MLP([self.dim, int(self.hidden_dims / 2), y_dim]))
        self.class_classifier.append(
            nn.ModuleList(
                [simple_MLP([self.dim, int(self.hidden_dims / 2), 1]) for _ in range(y_dim)]
            )
        )
    
    def set_parameters(self, data_id, cls, avg_model):
        self.class_classifier[data_id][cls].layers[0].weight.data = avg_model.weight1.clone().detach()
        self.class_classifier[data_id][cls].layers[2].weight.data = avg_model.weight2.clone().detach()
    
    def get_model_list(self, cls):
        model_list = []
        for temp_id in range(len(self.class_classifier) - 1):
            model_list.append(self.class_classifier[temp_id][0])
            model_list.append(self.class_classifier[temp_id][1])
        for temp_cls in range(cls + 1):
            model_list.append(self.class_classifier[-1][temp_cls])
        
        return model_list