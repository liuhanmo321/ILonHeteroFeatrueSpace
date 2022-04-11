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
        num_tasks = 3,
        dim_head = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        y_dim = 2,
        device = None
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

        # self.num_continuous = num_continuous
        self.num_continuous = [num_continuous]

        # extractor parameters
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.device = device

        # structure parameters

        self.hidden_dims = 128
        self.embed_dim = 32

        # start modifying embeddings

        self.embeds = nn.ModuleList([nn.Embedding(total_tokens, self.dim)])
        self.simple_MLP = nn.ModuleList(
                [nn.ModuleList(
                    [simple_MLP([1,self.embed_dim,self.dim]) for _ in range(num_continuous)]
                )]
            )
        
        self.gate = nn.Sigmoid()

        self.task_embeds = nn.ModuleList([nn.Embedding(num_tasks, self.dim) for _ in range(depth)])
        self.task_embeds.append(nn.Embedding(num_tasks, int(self.hidden_dims / 2)))
        
        # end modifying embeddings

        # start modification

        self.shared_extractor = nn.ModuleList(
            [singleTransformer(
                dim = dim,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            ) for _ in range(depth)]
        )

        self.shared_classifier_ly1 = nn.ModuleList([nn.Linear(dim, int(self.hidden_dims / 2)) for _ in range(num_tasks)])
        self.shared_classifier_ly2 = nn.ModuleList([nn.Linear(int(self.hidden_dims / 2), y_dim) for _ in range(num_tasks)])

        # end modification        
        
    def forward(self, x_categ, x_cont, data_id, s=1):
        masks = self.mask(data_id, s=s)
        # print(masks)    
        x = None   
        for i in range(self.depth):
            if i == 0:
                x = self.shared_extractor[i](x_categ, x_cont)
                # print(x.shape)
            else:
                x = self.shared_extractor[i](x)
            x = x * masks[i].view(1,1,-1).expand_as(x)
        
        x = x[:, 0, :]
        x = nn.ReLU()(self.shared_classifier_ly1[data_id](x))
        x = x * masks[self.depth].expand_as(x)
        x = self.shared_classifier_ly2[data_id](x)

        return x, masks

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
            
    def mask(self, data_id, s=1):
        data_id = torch.autograd.Variable(torch.LongTensor([data_id]), requires_grad=False).to(self.device)
        # print(data_id)
        # print(self.task_embeds[data_id.data[0]](data_id))
        gc = [self.gate(s * emb(data_id)) for emb in self.task_embeds]
        return gc
    

    def get_view_for(self,n,masks):
        if 'shared_extractor' in n:
            d = int(n.split('.')[1])
            if 'net.3.weight' in n:
                if d == 0:
                    post=masks[d].data.view(-1,1).expand_as(self.get_parameter(n))
                    return post
                else:
                    post=masks[d].data.view(-1,1).expand_as(self.get_parameter(n))
                    pre = masks[d-1].data.view(-1,1).expand_as(self.get_parameter(n))
                    return torch.min(post, pre)
            if 'net.3.bias' in n:
                return masks[d].data.view(-1)
        
        if 'shared_classifier_ly1' in n:
            if 'weight' in n:
                post = masks[self.depth].data.view(-1,1).expand_as(self.get_parameter(n))
                pre = masks[self.depth - 1].data.view(1,-1).expand_as(self.get_parameter(n))
                return torch.min(post, pre)
            else:
                return masks[self.depth].data.view(-1)

        return None