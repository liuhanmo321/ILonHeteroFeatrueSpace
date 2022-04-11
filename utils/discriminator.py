import torch

class Discriminator(torch.nn.Module):
    def __init__(self,params,task_id):
        super(Discriminator, self).__init__()

        # self.num_tasks=args.ntasks
        # self.units=args.units
        # self.latent_dim=args.latent_dim

        self.num_tasks= params['ntasks']
        self.units = 64
        self.latent_dim= params['num_shared_features']

        self.dis = torch.nn.Sequential(
            GradientReversal(),
            torch.nn.Linear(self.latent_dim, self.units),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.units, self.units),
            torch.nn.Linear(self.units, task_id + 2)
        )


    def forward(self, z, labels, task_id):
        return self.dis(z)

    def pretty_print(self, num):
        magnitude=0
        while abs(num) >= 1000:
            magnitude+=1
            num/=1000.0
        return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


    def get_size(self):
        count=sum(p.numel() for p in self.dis.parameters() if p.requires_grad)
        print('Num parameters in D       = %s ' % (self.pretty_print(count)))


class GradientReversalFunction(torch.autograd.Function):
    """
    From:
    https://github.com/jvanvugt/pytorch-domain-adaptation/blob/cb65581f20b71ff9883dd2435b2275a1fd4b90df/utils.py#L26
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)