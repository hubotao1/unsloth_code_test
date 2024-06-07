#  单独测试Triton f_partial和PyTorch f_partial中用到的每个小算子结果，以及最后的结果对比
# 注意此处如果输入是一个标量的话，结果测试是没有问题的；如果输入是tensor类型，则tensor的第二个
# 值结果会出现差异，主要原因是我这里设置的grid为1 ，如果按照n_elements = x.numel()
# grid = lambda opt: (triton.cdiv(n_elements, 1)   sqrt_kernel[grid](x, y)输入元素的数量来设置
# grid的值的话，结果测试就是完全正确的，已将下面的代码修正

#  单独测试Triton f_partial和PyTorch f_partial中用到的每个小算子结果，以及最后的结果对比
import torch
import triton
import triton.language as tl

# Triton kernel for sqrt
@triton.jit
def sqrt_kernel(x, y):
    idx = tl.program_id(0)
    x_val = tl.load(x + idx).to(tl.float32)
    y_val = tl.math.sqrt(x_val)
    tl.store(y + idx, y_val)

# Triton kernel for rsqrt
@triton.jit
def rsqrt_kernel(x, y):
    idx = tl.program_id(0)
    x_val = tl.load(x + idx).to(tl.float32)
    y_val = tl.math.rsqrt(x_val)
    tl.store(y + idx, y_val)

# Triton kernel for erf
@triton.jit
def erf_kernel(x, y):
    idx = tl.program_id(0)
    x_val = tl.load(x + idx).to(tl.float32)
    y_val = tl.math.erf(x_val)
    tl.store(y + idx, y_val)

# Triton kernel for f_partial
@triton.jit
def f_partial_kernel(x, y):
    idx = tl.program_id(0)
    x_val = tl.load(x + idx).to(tl.float32)
    y_val = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * x_val) + 1.0)
    tl.store(y + idx, y_val)

# Input tensor
e = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')

# Compare sqrt
torch_sqrt = torch.sqrt(e)
print("PyTorch sqrt:", torch_sqrt)

x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')

y = torch.empty_like(x)
# 获取张量的元素数量
n_elements = x.numel()
    
# Launch kernel
grid = lambda opt: (triton.cdiv(n_elements, 1),)
sqrt_kernel[grid](x, y)
print("Triton sqrt:", y)

# Compare rsqrt
torch_rsqrt = torch.rsqrt(e)
print("PyTorch rsqrt:", torch_rsqrt)

x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')

y = torch.empty_like(x)
# 获取张量的元素数量
n_elements = x.numel()
    
# Launch kernel
grid = lambda opt: (triton.cdiv(n_elements, 1),)
rsqrt_kernel[grid](x, y)

print("Triton rsqrt:", y)

# Compare erf
torch_erf = torch.erf(e)
print("PyTorch erf:", torch_erf)

x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')
y = torch.empty_like(x)
# 获取张量的元素数量
n_elements = x.numel()
    
# Launch kernel
grid = lambda opt: (triton.cdiv(n_elements, 1),)
erf_kernel[grid](x, y)
print("Triton erf:", y)

# Compare f_partial
torch_f_partial = 0.5 * (torch.erf(torch.rsqrt(torch.tensor(2.0, device='cuda')) * e) + 1.0)
print("PyTorch f_partial:", torch_f_partial)

x = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
                    [0.3074, 0.6341, 0.4901, 0.8964],
                    [0.4556, 0.6323, 0.3489, 0.4017],
                    [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')
y = torch.empty_like(x)
# 获取张量的元素数量
n_elements = x.numel()
    
# Launch kernel
grid = lambda opt: (triton.cdiv(n_elements, 1),)
f_partial_kernel[grid](x, y)
print("Triton f_partial:", y)




# import torch
# import triton
# import triton.language as tl

# # Triton kernel for rsqrt
# @triton.jit
# def rsqrt_kernel(x, y):
#     idx = tl.program_id(0)
#     x_val = tl.load(x + idx).to(tl.float32)
#     y_val = tl.math.rsqrt(x_val)
#     tl.store(y + idx, y_val)

# # Triton kernel for erf
# @triton.jit
# def erf_kernel(x, y):
#     idx = tl.program_id(0)
#     x_val = tl.load(x + idx).to(tl.float32)
#     y_val = tl.math.erf(x_val)
#     tl.store(y + idx, y_val)

# # Triton kernel for f_partial
# @triton.jit
# def f_partial_kernel(x, y):
#     idx = tl.program_id(0)
#     x_val = tl.load(x + idx).to(tl.float32)
#     y_val = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * x_val) + 1.0)
#     tl.store(y + idx, y_val)

# # Input tensor
# e = torch.tensor([2.0], dtype=torch.float32, device='cuda')

# # Compare rsqrt
# torch_rsqrt = torch.rsqrt(e)
# print("PyTorch rsqrt:", torch_rsqrt)

# x = torch.tensor([2.0], dtype=torch.float32, device='cuda')
# y = torch.empty_like(x)
# rsqrt_kernel[(1,)](x, y)
# print("Triton rsqrt:", y)

# # Compare erf
# torch_erf = torch.erf(e)
# print("PyTorch erf:", torch_erf)

# x = torch.tensor([2.0], dtype=torch.float32, device='cuda')
# y = torch.empty_like(x)
# erf_kernel[(1,)](x, y)
# print("Triton erf:", y)

# # Compare f_partial
# torch_f_partial = 0.5 * (torch.erf(torch.rsqrt(torch.tensor(2.0, device='cuda')) * e) + 1.0)
# print("PyTorch f_partial:", torch_f_partial)

# x = torch.tensor([2.0], dtype=torch.float32, device='cuda')
# y = torch.empty_like(x)
# f_partial_kernel[(1,)](x, y)
# print("Triton f_partial:", y)

