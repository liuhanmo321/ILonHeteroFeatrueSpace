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
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col',
        final_mlp_style = 'common',
        y_dim = 2,
        num_tasks = 3
        ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # create category embeddings table

        total_tokens = sum(categories)

        # self.total_tokens = sum(categories)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        # categories_offset = torch.unsqueeze(categories_offset, 0)
        print('categorical_offset', categories_offset)
        
        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        # self.num_continuous = [num_continuous]

        # extractor parameters
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout

        # structure parameters
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        self.embeds = nn.Embedding(total_tokens, self.dim)

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(num_continuous)])
        else:
            print('Continous features are not passed through attention') 

        # start modifying embeddings

        # self.embeds = nn.ModuleList([nn.Embedding(total_tokens, self.dim)])
        # self.simple_MLP = nn.ModuleList(
        #         [nn.ModuleList(
        #             [simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)]
        #         )]
        #     )
        
        # end modifying embeddings

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )

        # start modification
        self.shared_extractor = Transformer(
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )

        self.specific_extractor = nn.ModuleList(
                [Transformer(
                        dim = dim,
                        depth = depth,
                        heads = heads,
                        dim_head = dim_head,
                        attn_dropout = attn_dropout,
                        ff_dropout = ff_dropout
                )]
            )

        self.shared_classifier = nn.ModuleList([simple_MLP([dim ,1000, y_dim])])
        self.specific_classifier = nn.ModuleList([simple_MLP([dim ,1000, y_dim])])

        self.embeddings = nn.ModuleList(
            [nn.Embedding(total_tokens, self.dim)]
        )

        # end modification        

        self.mlpfory = simple_MLP([dim ,1000, y_dim])

        
    def forward(self, x_categ, x_cont):        
        x = self.transformer(x_categ, x_cont)
        return x

    def add_task(self, categories, num_continuous, y_dim):
        total_tokens = sum(categories)
        self.num_continuous.append(num_continuous)

        temp_categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = 0)
        temp_categories_offset = temp_categories_offset.cumsum(dim = -1)[:-1]
        self.categories_offset = torch.cat((self.categories_offset, temp_categories_offset), dim=0)

        self.embeds.append(nn.Embedding(total_tokens, self.dim))
        self.simple_MLP.append(
            nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous[-1])])
        )
        self.specific_extractor.append(Transformer(
                        dim = self.dim,
                        depth = self.depth,
                        heads = self.heads,
                        dim_head = self.dim_head,
                        attn_dropout = self.attn_dropout,
                        ff_dropout = self.ff_dropout
                ))
        self.shared_classifier.append(simple_MLP([self.dim ,1000, y_dim]))
        self.specific_classifier.append(simple_MLP([self.dim ,1000, y_dim]))
