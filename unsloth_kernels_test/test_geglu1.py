# 前向测试通过，反向f_partial_row的值和torch计算的f_partial不一样，但单独拎出来算结果一样
import torch
from geglu import geglu_exact_forward_kernel, geglu_exact_backward_kernel, geglu_approx_forward_kernel, geglu_approx_backward_kernel

# 设置随机种子
torch.manual_seed(42)

# 精确前向传播
def exact_forward_kernel(e,g):
    f = 0.5 * e * (torch.erf(e / torch.sqrt(torch.tensor(2.0))) + 1.0)
    h = f * g
    return h

# 精确反向传播
def exact_backward_kernel(DW , e, g):
    # f_partial = 0.5 * (torch.erf(torch.rsqrt(torch.tensor(2.0)) * e) + 1.0)
    f_partial = 0.5 * (torch.erf(torch.rsqrt(torch.tensor(2.0, device='cuda')) * e) + 1.0)
    f = f_partial * e
    h = f * g
    df = DW * f
    dg = DW * g
    t = 0.3989422804014327 # 常数 1/sqrt(2*pi) 的值
    df_de_ = f_partial + t * e * torch.exp(-0.5 * e * e)
    de = dg * df_de_
    return h, df, de, f_partial

# 近似前向传播
def approx_forward_kernel(e,g):
    s = 0.7978845608028654 # math.sqrt(2 / math.pi)
    f = 0.5 * e * (torch.tanh(s * e * (1 + 0.044715 * e * e)) + 1.0)
    h = f * g
    return h

# 近似反向传播
def approx_backward_kernel(DW, e, g):
    s = 0.7978845608028654 # math.sqrt(2 / math.pi)
    a = s * e
    b = a * 0.044715 * e * e
    T = 1.0 + torch.tanh(a + b)
    T2 = 0.5 * T
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2
    f = T2 * e
    h = f * g
    df = DW * f
    dg = DW * g
    de = dg * df_de
    return h, df, de

def test_geglu_kernels():
    batch, seq_len, hd = 1, 4, 4

    # 固定输入值
    e = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                       [0.3074, 0.6341, 0.4901, 0.8964],
                       [0.4556, 0.6323, 0.3489, 0.4017],
                       [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')

    g = torch.tensor([[[0.6977, 0.8000, 0.1610, 0.2823],
                       [0.6816, 0.9152, 0.3972, 0.8742],
                       [0.4599, 0.0558, 0.3775, 0.9152],
                       [0.4474, 0.1633, 0.0737, 0.0021]]], dtype=torch.float32, device='cuda')

    DW = torch.tensor([[[0.3461, 0.1853, 0.1745, 0.7490],
                        [0.8124, 0.9171, 0.7106, 0.4795],
                        [0.9182, 0.4614, 0.4951, 0.4182],
                        [0.2345, 0.5412, 0.5010, 0.6936]]], dtype=torch.float32, device='cuda')

    print("e:", e)
    print("g:", g)
    print("DW:", DW)

    # 精确前向传播测试
    h_triton = geglu_exact_forward_kernel(e, g)
    h_pytorch = exact_forward_kernel(e, g)
    print("Triton Exact Forward:", h_triton)
    print("PyTorch Exact Forward:", h_pytorch)
    assert torch.allclose(h_triton, h_pytorch, atol=1e-5), "Exact forward kernel test failed"
    print("精确前向传播测试 passed!")

    # 精确反向传播测试
    h_triton, df_triton, de_triton, f_partial_triton = geglu_exact_backward_kernel(DW, e, g)
    h_pytorch, df_pytorch, de_pytorch, f_partial_pytorch = exact_backward_kernel(DW, e, g)
    print("Triton Exact Backward h:", h_triton)
    print("PyTorch Exact Backward h:", h_pytorch)
    print("Triton Exact Backward df:", df_triton)
    print("PyTorch Exact Backward df:", df_pytorch)
    print("Triton Exact Backward de:", de_triton)
    print("PyTorch Exact Backward de:", de_pytorch)
    print("Triton Exact Backward f_partial:", f_partial_triton)
    print("PyTorch Exact Backward f_partial:", f_partial_pytorch)
    assert torch.allclose(h_triton, h_pytorch, atol=1e-5), "Exact backward kernel h test failed"
    assert torch.allclose(df_triton, df_pytorch, atol=1e-5), "Exact backward kernel df test failed"
    assert torch.allclose(de_triton, de_pytorch, atol=1e-5), "Exact backward kernel de test failed"
    assert torch.allclose(f_partial_triton, f_partial_pytorch, atol=1e-5), "Exact backward kernel f_partial test failed"
    print("精确反向传播测试 passed!")

    # 近似前向传播测试
    h_triton = geglu_approx_forward_kernel(e, g)
    h_pytorch = approx_forward_kernel(e, g)
    print("Triton Approx Forward:", h_triton)
    print("PyTorch Approx Forward:", h_pytorch)
    assert torch.allclose(h_triton, h_pytorch, atol=1e-5), "Approx forward kernel test failed"
    print("近似前向传播测试 passed!")

    # 近似反向传播测试
    h_triton, df_triton, de_triton = geglu_approx_backward_kernel(DW, e, g)
    h_pytorch, df_pytorch, de_pytorch = approx_backward_kernel(DW, e, g)
    print("Triton Approx Backward h:", h_triton)
    print("PyTorch Approx Backward h:", h_pytorch)
    print("Triton Approx Backward df:", df_triton)
    print("PyTorch Approx Backward df:", df_pytorch)
    print("Triton Approx Backward de:", de_triton)
    print("PyTorch Approx Backward de:", de_pytorch)
    assert torch.allclose(h_triton, h_pytorch, atol=1e-5), "Approx backward kernel h test failed"
    assert torch.allclose(df_triton, df_pytorch, atol=1e-5), "Approx backward kernel df test failed"
    assert torch.allclose(de_triton, de_pytorch, atol=1e-5), "Approx backward kernel de test failed"
    print("近似反向传播测试 passed!")

if __name__ == "__main__":
    test_geglu_kernels()
