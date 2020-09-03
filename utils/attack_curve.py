import torch
import torch.nn.functional as F
import torch.nn as nn


class AttackPGD(nn.Module):
    def __init__(self, basic_net, config, loss_func=None):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.device = basic_net.device
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.device)
        mean = torch.reshape(mean, [3, 1, 1])
        std = torch.tensor([0.247, 0.243, 0.261], device=self.device)
        std = torch.reshape(std, [3, 1, 1])
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'
        if loss_func is not None:
            self.loss_func = loss_func

    def forward(self, inputs, targets, t=None, norm=False):
        if hasattr(self, 'loss_func'):
            loss_func = self.loss_func
        else:
            loss_func = F.cross_entropy  
        x = inputs.detach()
        if t is None:
            t = inputs.data.new(1).uniform_(0.0, 1.0)
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                if norm: 
                    x_norm = (x - self.mean) / self.std
                else:
                    x_norm = x 
                logits = self.basic_net(x_norm, t)
                loss = loss_func(logits, targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        if norm: 
            x_norm = (x - self.mean) / self.std
        else:
            x_norm = x 
        x_out = self.basic_net(x_norm, t)
        return x_out, x_norm
