from .model import *

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
        num_side_classifier = 3
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

        # structure parameters

        self.hidden_dims = 128
        self.embed_dim = 32
        self.num_side_classifier = num_side_classifier

        # start modifying embeddings

        self.embeds = nn.ModuleList([nn.Embedding(total_tokens, self.dim)])
        self.simple_MLP = nn.ModuleList(
                [nn.ModuleList(
                    [simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)]
                )]
            )
        
        self.unlabeled_embeds = nn.ModuleList([])
        self.unlabeled_simple_MLP = nn.ModuleList([])
        
        # end modifying embeddings

        # start modification

        self.shared_extractor = Transformer(
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        self.shared_classifier = nn.ModuleList([simple_MLP([dim, int(self.hidden_dims / 2), y_dim])])
        self.side_classifier = nn.ModuleList(
            [nn.ModuleList(
                [simple_MLP([dim, int(self.hidden_dims / 2), y_dim]) for _ in range(self.num_side_classifier)]
            )]
        )

        # end modification        
        
    def forward(self, x_categ, x_cont):        
        x = self.shared_extractor(x_categ, x_cont)
        return x

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
        
        self.shared_classifier.append(simple_MLP([self.dim, int(self.hidden_dims / 2), y_dim]))
        self.side_classifier.append(
            nn.ModuleList([simple_MLP([self.dim, int(self.hidden_dims / 2), y_dim]) for _ in range(self.num_side_classifier)])
        )
    
    def add_unlabeled_task(self, categories, num_continuous, y_dim):
        total_tokens = sum(categories)
        self.unlabeled_num_continuous.append(num_continuous)

        temp_categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        temp_categories_offset = temp_categories_offset.cumsum(dim = -1)[:-1]
        self.unlabeled_categories_offset.append(temp_categories_offset)

        print('categorical_offset', self.unlabeled_categories_offset)

        self.unlabeled_embeds.append(nn.Embedding(total_tokens, self.dim))
        self.unlabeled_simple_MLP.append(
            nn.ModuleList([simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)])
        )