# 测试通过
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorch_RMS_Layernorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(PyTorch_RMS_Layernorm, self).__init__()
        # 创建一个形状为 normalized_shape 的权重参数 weight，其初始值为 1，并注册为模型的参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, X):
        row_var = torch.mean(X * X, dim=-1, keepdim=True)
        inv_var = torch.rsqrt(row_var + self.eps)
        normed = X * inv_var
        return normed * self.weight

    def backward(self, dY, X, inv_var):
        n_cols = X.shape[-1]
        normed = X * inv_var
        dY_W = dY * self.weight

        rowsum_dY_normed = torch.sum(dY_W * normed, dim=-1, keepdim=True)
        dX = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
        return dX

# Test the PyTorch implementation
def test_rms_layernorm():
    torch.manual_seed(42)
    
    # Parameters
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    eps = 1e-6
    
    # Input tensors
    X = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', requires_grad=True)
    W = torch.randn(hidden_dim, device='cuda', requires_grad=True)

    # PyTorch RMS Layernorm
    layernorm = PyTorch_RMS_Layernorm(hidden_dim, eps=eps).to('cuda')
    # 将 layernorm 的权重参数 weight 设置为 W 的克隆,确保不会影响原始的 W 权重
    layernorm.weight.data = W.clone() 
    pytorch_Y = layernorm(X)
    
    # Triton RMS Layernorm
    from rms_layernorm import fast_rms_layernorm, Fast_RMS_Layernorm
    
    triton_Y = fast_rms_layernorm(layernorm, X, gemma=False)
    
    # Compare forward pass results
    print("PyTorch forward output:\n", pytorch_Y)
    print("Triton forward output:\n", triton_Y)
    assert torch.allclose(pytorch_Y, triton_Y, atol=1e-5), "forward kernel h test failed"
    print("forward kernel h test passed!")

    # Backward pass  先输入生成一个与 X 形状相同、并且包含标准正态分布随机数的新张量 dY
    # 原本dY应该由损失函数计算得到，但此处triton源码中无损失函数，所以先给个dY值用以测试
    dY = torch.randn_like(X, device='cuda')
    
    # PyTorch backward
    inv_var = 1.0 / torch.sqrt(torch.mean(X * X, dim=-1, keepdim=True) + eps)
    dX_pytorch = layernorm.backward(dY, X, inv_var)
    
    # Triton backward
    Y = Fast_RMS_Layernorm.apply(X, W, eps, False)
    Y.backward(dY)
    dX_triton = X.grad
    
    # Compare backward pass results
    print("PyTorch backward output:\n", dX_pytorch)
    print("Triton backward output:\n", dX_triton)
    assert torch.allclose(dX_pytorch, dX_triton, atol=1e-5), "backward kernel h test failed"
    print("backward kernel h test passed!")

if __name__ == "__main__":
    test_rms_layernorm()
