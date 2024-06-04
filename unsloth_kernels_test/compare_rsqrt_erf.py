import torch
import triton
import triton.language as tl

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
e = torch.tensor([2.0], dtype=torch.float32, device='cuda')

# Compare rsqrt
torch_rsqrt = torch.rsqrt(e)
print("PyTorch rsqrt:", torch_rsqrt)

x = torch.tensor([2.0], dtype=torch.float32, device='cuda')
y = torch.empty_like(x)
rsqrt_kernel[(1,)](x, y)
print("Triton rsqrt:", y)

# Compare erf
torch_erf = torch.erf(e)
print("PyTorch erf:", torch_erf)

x = torch.tensor([2.0], dtype=torch.float32, device='cuda')
y = torch.empty_like(x)
erf_kernel[(1,)](x, y)
print("Triton erf:", y)

# Compare f_partial
torch_f_partial = 0.5 * (torch.erf(torch.rsqrt(torch.tensor(2.0, device='cuda')) * e) + 1.0)
print("PyTorch f_partial:", torch_f_partial)

x = torch.tensor([2.0], dtype=torch.float32, device='cuda')
y = torch.empty_like(x)
f_partial_kernel[(1,)](x, y)
print("Triton f_partial:", y)


# import torch
# import triton
# import triton.language as tl

# # PyTorch计算f_partial
# e = torch.tensor([[[0.4963, 0.7682, 0.0885, 0.1320],
#                    [0.3074, 0.6341, 0.4901, 0.8964],
#                    [0.4556, 0.6323, 0.3489, 0.4017],
#                    [0.0223, 0.1689, 0.2939, 0.5185]]], dtype=torch.float32, device='cuda')

# f_partial = 0.5 * (torch.erf(torch.rsqrt(torch.tensor(2.0, device='cuda')) * e) + 1.0)
# print("PyTorch f_partial:", f_partial)

# # Triton计算f_partial_row
# @triton.jit
# def f_partial_kernel(x, y):
#     idx = tl.program_id(0)
#     x_val = tl.load(x + idx).to(tl.float32)
#     y_val = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * x_val) + 1.0)
#     tl.store(y + idx, y_val)

# # Output tensor
# f_partial_triton = torch.empty_like(e)

# # Launch Triton kernel
# n_elements = e.numel()
# f_partial_kernel[(n_elements,)](e.flatten(), f_partial_triton.flatten())

# print("Triton f_partial_row:", f_partial_triton)
