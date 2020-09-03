import torch
import torch.nn.functional as F
import torch.nn as nn


class AttackPGD(nn.Module):
    def __init__(self, basic_net, config, loss_func=F.cross_entropy, optimizer=None):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.device = basic_net.device
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = loss_func
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.device)
        mean = torch.reshape(mean, [3, 1, 1])
        std = torch.tensor([0.247, 0.243, 0.261], device=self.device)           
        std = torch.reshape(std, [3, 1, 1])  
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        if optimizer is not None:
            self.optimizer = optimizer 

    def forward(self, inputs, targets, norm=False, amp_flag=False):
        x = inputs.detach()

        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                if norm: 
                    x_norm = (x - self.mean) / self.std 
                else:
                    x_norm = x
                logits = self.basic_net(x_norm)
                loss = self.loss_func(logits, targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
            if hasattr(self, 'optimizer') and self.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()           
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        if norm: 
            x_norm = (x - self.mean) / self.std 
        else:
            x_norm = x 
        x_out = self.basic_net(x_norm)
        return x_out, x_norm
