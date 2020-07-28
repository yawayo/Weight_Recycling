import math
import time
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.autograd import Variable

class Weight_recycle_Conv2d(Module):

    def __init__(self, in_features, out_features, kernel_size, padding=0, bias=True):
        super(Weight_recycle_Conv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_features//4, in_features, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features//4))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight = torch.nn.init.kaiming_normal_(self.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight_0 = self.weight.data
        weight_90 = torch.rot90(weight_0, 1, [2, 3])
        weight_180 = torch.rot90(weight_90, 1, [2, 3])
        weight_270 = torch.rot90(weight_180, 1, [2, 3])
        weight_all = torch.cat((weight_0, weight_90, weight_180, weight_270), dim=0)
        bias_all = torch.cat((self.bias, self.bias, self.bias, self.bias))
        return F.conv2d(input, weight=weight_all, bias=bias_all, padding=self.padding)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', bias=' + str(self.bias is not None) + ')'


'''
class weight_recycle_convolution(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return F.conv2d(input, weight, bias)
        """
        batch_size, C_in, H_in, W_in = input.shape
        output_C, input_C, K_H, K_W = weight.shape
        H_out = (H_in + 2*padding - K_H) + 1
        W_out = (H_in + 2*padding - K_W) + 1
        output = torch.empty(batch_size, output_C, H_out, W_out).to("cuda:3")

        padding = (padding, padding, padding, padding)
        input = F.pad(input, padding, 'constant', 0)
        for b in range(batch_size):
            for o in range(output_C):
                for h in range(H_out):
                    for w in range(W_out):
                        conv_mul = input[b][:][:, h:h+3][:, :, w:w+3] * weight[o]
                        conv_sum = torch.sum(conv_mul)
                        if bias is not None:
                            conv_sum = torch.sum(conv_mul) +  bias[o]
                        else:
                            conv_sum = torch.sum(conv_mul)
                        output[b][o][h][w] = conv_sum
            print(b)

        return output
        """
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        print("grad_output", grad_output.shape)
        print("input", input.shape)
        print("weight", weight.shape)
        print("bias", bias.shape)
        time.sleep(1)
        return grad_input, grad_weight, grad_bias
        """
        print("grad_output", grad_output.shape)
        print()
        print("back_input", input.shape)
        print("back_weight", weight.shape)
        print("back_bias", bias.shape)
        print()
        grad_input_np = torch.zeros([grad_output.size(0), grad_output.size(1)])
        print("grad_input_np", grad_input_np.shape)
        print("weight.data[0]", weight.data[0].shape)
        print()
        for i in range(input.size(0)):
            grad_input_np[i] = weight.data[0]
        grad_input = Variable(torch.from_numpy(grad_input_np))
        print("grad_input", grad_input.shape)

        grad_weight_np = np.zeros([1], dtype=np.float32)
        for i in range(input.size(0)):
            grad_weight_np[0] += input.data[i][0] * grad_output.data[i][0]
        grad_weight = Variable(torch.from_numpy(grad_weight_np))

        grad_bias_np = np.zeros([1], dtype=np.float32)
        for i in range(input.size(0)):
            grad_bias_np[0] += grad_output.data[i][0]
        grad_bias = Variable(torch.from_numpy(grad_bias_np))

        return grad_input, grad_weight, grad_bias
        """
        """
        input, weight, bias = ctx.saved_variables
        print("back_input", input.shape)
        print("back_weight", weight.shape)
        print("back_bias", bias.shape)
        grad_input = grad_weight = grad_bias = None
        print("grad_output", grad_output.shape)
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output)
            print("grad_input", grad_input.shape)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output)
            print("grad_weight", grad_weight.shape)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
            print("grad_bias", grad_bias.shape)

        return grad_input, grad_weight, grad_bias
        
        """


class Weight_recycle_Conv2d(Module):

    def __init__(self, in_features, out_features, kernel_size, padding=0, bias=True):
        super(Weight_recycle_Conv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_features, in_features, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight = torch.nn.init.kaiming_normal_(self.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.padding == 0:
            return weight_recycle_convolution.apply(input, self.weight, self.bias)
        else:
            padding_size = (self.padding, self.padding, self.padding, self.padding)
            input = F.pad(input, padding_size, 'constant', 0)
            return weight_recycle_convolution.apply(input, self.weight, self.bias)
        #return F.conv2d(input, self.weight, bias=self.bias, padding=self.padding)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

'''

"""

class Weight_recycle_Conv2d(Module):

    def __init__(self, in_features, out_features, kernel_size, padding, bias=True):
        super(Weight_recycle_Conv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = Parameter(torch.Tensor(out_features//4, in_features, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight = torch.nn.init.kaiming_normal_(self.weight)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight_0 = self.weight.data.cpu()
        weight_90 = torch.rot90(weight_0, 1, [2, 3])
        weight_180 = torch.rot90(weight_90, 1, [2, 3])
        weight_270 = torch.rot90(weight_180, 1, [2, 3])
        weight_all = torch.cat((weight_0, weight_90, weight_180, weight_270), dim=0).to("cuda")
        '''
        print("weight_0 : ", weight_0[0][0][0][0])
        print("weight_90 : ", weight_90[0][0][-1][0])
        print("weight_180 : ", weight_180[0][0][-1][-1])
        print("weight_270 : ", weight_270[0][0][0][-1])
        '''
        '''
        weight_90 = weight_0.transpose(2, 3).flip(3)
        weight_180 = weight_90.transpose(2, 3).flip(3)
        weight_270 = weight_180.transpose(2, 3).flip(3)
        weight_all = torch.cat((weight_0, weight_90, weight_180, weight_270), dim=0).to("cuda")
        '''

        return F.conv2d(input, weight_all, bias = self.bias, padding=self.padding)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
"""