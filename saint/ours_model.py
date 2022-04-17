from .model import *

def generate_embeddings():
    return False


class sep_MLP(nn.Module):
    def __init__(self,dim,len_feats,categories):
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            self.layers.append(simple_MLP([dim,5*dim, categories[i]]))

        
    def forward(self, x):
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:,i,:]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred


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
        condition = None
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
        
        # self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        # self.num_continuous = num_continuous
        self.num_continuous = [num_continuous]

        # extractor parameters
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        # structure parameters
        self.condition = condition

        self.hidden_dims = 128
        self.embed_dim = 32

        # start modifying embeddings

        self.embeds = nn.ModuleList([nn.Embedding(total_tokens, self.dim)])
        self.simple_MLP = nn.ModuleList(
                [nn.ModuleList(
                    [simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)]
                )]
            )
        
        # end modifying embeddings

        # start modification
        if condition != 'specific_only':
            self.shared_extractor = Transformer(
                    dim = dim,
                    depth = depth,
                    heads = heads,
                    dim_head = dim_head,
                    attn_dropout = attn_dropout,
                    ff_dropout = ff_dropout
                )
            self.shared_classifier = nn.ModuleList([simple_MLP([dim, int(self.hidden_dims / 2), y_dim])])

        if condition != 'shared_only':
            self.specific_extractor = nn.ModuleList(
                    [Transformer(
                            dim = dim,
                            depth = int(depth / 2),
                            # depth = 1,
                            heads = heads,
                            dim_head = dim_head,
                            attn_dropout = attn_dropout,
                            ff_dropout = ff_dropout
                    )]
                )
            self.specific_classifier = nn.ModuleList([simple_MLP([dim, int(self.hidden_dims / 2), y_dim])])       

        # end modification        
        
    def forward(self, x_categ, x_cont):        
        x = self.shared_extractor(x_categ, x_cont)
        return x

    def add_task(self, categories, num_continuous, y_dim):
        total_tokens = sum(categories)
        self.num_continuous.append(num_continuous)

        temp_categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        temp_categories_offset = temp_categories_offset.cumsum(dim = -1)[:-1]
        # temp_categories_offset = torch.unsqueeze(temp_categories_offset, 0)
        # self.categories_offset = torch.cat((self.categories_offset, temp_categories_offset), dim=0)
        self.categories_offset.append(temp_categories_offset)

        print('categorical_offset', self.categories_offset)

        self.embeds.append(nn.Embedding(total_tokens, self.dim))
        self.simple_MLP.append(
            nn.ModuleList([simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)])
        )

        if self.condition != 'shared_only':
            self.specific_extractor.append(Transformer(
                            dim = self.dim,
                            depth = int(self.depth / 2),
                            # depth = 1,
                            heads = self.heads,
                            dim_head = self.dim_head,
                            attn_dropout = self.attn_dropout,
                            ff_dropout = self.ff_dropout
                    ))
            self.specific_classifier.append(simple_MLP([self.dim, int(self.hidden_dims / 2), y_dim]))
        
        if self.condition != 'specific_only':
            self.shared_classifier.append(simple_MLP([self.dim, int(self.hidden_dims / 2), y_dim]))
        
