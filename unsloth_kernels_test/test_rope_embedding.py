# 测试全部通过
import torch
import torch.nn.functional as F
from rope_embedding import Slow_RoPE_Embedding, Fast_RoPE_Embedding

class PyTorchSlowRoPEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin, position_ids=None):
        if position_ids is not None:
            cos = cos.squeeze(0).squeeze(0)  # [seq_len, dim]
            sin = sin.squeeze(0).squeeze(0)  # [seq_len, dim]
            cos = cos[position_ids]  # [seq_len, dim]
            sin = sin[position_ids]  # [seq_len, dim]
            cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
            sin = sin.unsqueeze(1)  # [bs, 1, seq_len, dim]

        half = Q.shape[-1] // 2
        RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
        Q = Q * cos + RH_Q * sin
        ctx.save_for_backward(cos, sin)
        return Q

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        half = dY.shape[-1] // 2
        RH_dY = torch.cat((dY[..., half:], -dY[..., :half]), dim=-1)
        dY = dY * cos + RH_dY * sin
        return dY, None, None, None

class PyTorchFastRoPEEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        batch, seq_len, n_heads, head_dim = Q.shape
        half_head_dim = head_dim // 2

        Q = Q.view(batch * seq_len, n_heads, head_dim)
        n_rows, _, _ = Q.shape

        Q_result = torch.zeros_like(Q)

        for row_position in range(n_rows):
            for head in range(n_heads):
                # Calculate the offsets for Q1 and Q2
                # :half_head_dim  表示最后一个维度（head_dim）中的前半部分
                # 从 Q 中提取指定位置和头的前半部分，得到一个形状为 (half_head_dim,) 的张量
                Q1 = Q[row_position, head, :half_head_dim]
                # half_head_dim: 表示最后一个维度（head_dim）中的后半部分
                # 从 Q 中提取指定位置和头的后半部分，得到一个形状为 (half_head_dim,) 的张量
                Q2 = Q[row_position, head, half_head_dim:]

                # sin1 和 cos1 是从 sin 和 cos 张量中提取的对应位置的正弦和余弦值。
                # 使用 row_position % seq_len 计算出对应的序列位置，并提取前半部分的正弦和余弦值。
                sin1 = sin[row_position % seq_len, :half_head_dim]
                cos1 = cos[row_position % seq_len, :half_head_dim]

                # 实现了 RoPE 变换，将前半部分和后半部分分别与 cos1 和 sin1 进行加权组合。
                # 计算前半部分的 RoPE 变换结果，存储在 Q_result 的对应位置。
                Q_result[row_position, head, :half_head_dim] = Q1 * cos1 - Q2 * sin1 
                # 计算后半部分的 RoPE 变换结果，存储在 Q_result 的对应位置。
                Q_result[row_position, head, half_head_dim:] = Q2 * cos1 + Q1 * sin1

        Q_result = Q_result.view(batch, seq_len, n_heads, head_dim)

        ctx.save_for_backward(cos, sin)
        return Q_result

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        batch, seq_len, n_heads, head_dim = dY.shape
        dY = dY.view(batch * seq_len, n_heads, head_dim)
        n_rows, _, _ = dY.shape

        half_head_dim = head_dim // 2

        dY_result = torch.zeros_like(dY)

        for row_position in range(n_rows):
            for head in range(n_heads):

                dY1 = dY[row_position, head, :half_head_dim]
                dY2 = dY[row_position, head, half_head_dim:]

                sin1 = -(sin[row_position % seq_len, :half_head_dim])
                cos1 = cos[row_position % seq_len, :half_head_dim]

                dY_result[row_position, head, :half_head_dim] = dY1 * cos1 - dY2 * sin1
                dY_result[row_position, head, half_head_dim:] = dY2 * cos1 + dY1 * sin1

        dY_result = dY_result.view(batch, seq_len, n_heads, head_dim)
        return dY_result, None, None

def test_slow_rope_embedding():
    Q = torch.randn(2, 3, 4, 6, device="cuda", requires_grad=True)
    cos = torch.cos(torch.randn(3, 6, device="cuda"))  # [seq_len, dim]
    sin = torch.sin(torch.randn(3, 6, device="cuda"))  # [seq_len, dim]
    # 生成一个包含从 0 到 2 的整数序列的张量，即 [0, 1, 2]，1维张量，表示序列中的位置索引
    position_ids = torch.arange(0, 3, device="cuda")

    Q_slow = Slow_RoPE_Embedding.apply(Q.clone(), cos.clone(), sin.clone(), position_ids)
    Q_triton_slow_mean = Q_slow.mean()

    Q_pytorch = PyTorchSlowRoPEEmbedding.apply(Q.clone(), cos.clone(), sin.clone(), position_ids)
    Q_pytorch_loss_mean = Q_pytorch.mean()

    assert torch.allclose(Q_slow, Q_pytorch), "Slow_RoPE_Embedding outputs do not match!"
    print("Slow_RoPE_Embedding outputs match!")

    Q_triton_slow_mean.backward()
    triton_slow_grad = Q.grad.clone()
    Q.grad.zero_()

    Q_pytorch_loss_mean.backward()
    pytorch_slow_grad = Q.grad.clone()

    assert torch.allclose(triton_slow_grad, pytorch_slow_grad), "Slow_RoPE_Embedding backward pass does not match!"

    print("Slow_RoPE_Embedding outputs and backward pass match!")


def test_fast_rope_embedding():
    Q = torch.randn(2, 3, 4, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    cos = torch.cos(torch.randn(3, 6, device="cuda", dtype=torch.float32))  # [seq_len, dim]
    sin = torch.sin(torch.randn(3, 6, device="cuda", dtype=torch.float32))  # [seq_len, dim]

    Q_fast_triton = Fast_RoPE_Embedding.apply(Q.clone(), cos.clone(), sin.clone())
    Q_fast_triton_mean = Q_fast_triton.mean()

    Q_fast_pytorch = PyTorchFastRoPEEmbedding.apply(Q.clone(), cos.clone(), sin.clone())
    Q_fast_pytorch_mean = Q_fast_pytorch.mean()

    # print("Q_fast_triton output:", Q_fast_triton)
    # print("Q_fast_pytorch output:", Q_fast_pytorch)
    
    if not torch.allclose(Q_fast_triton, Q_fast_pytorch, atol=1e-5):
        diff = Q_fast_triton - Q_fast_pytorch
        print("Difference:", diff)
        print("Max difference:", torch.max(torch.abs(diff)))

    assert torch.allclose(Q_fast_triton, Q_fast_pytorch, atol=1e-5), "fast_rope_embedding test failed!"
    print("Fast_RoPE_Embedding outputs match!")

    
    Q_fast_triton_mean.backward()
    Q_fast_triton_grad = Q.grad.clone()
    Q.grad.zero_()

    Q_fast_pytorch_mean.backward()
    fast_pytorch_grad = Q.grad.clone()

    print("Q_fast_triton_grad output:", Q_fast_triton_grad)
    print("fast_pytorch_grad output:", fast_pytorch_grad)

    assert torch.allclose(Q_fast_triton_grad, fast_pytorch_grad, atol=1e-5), "Fast_RoPE_Embedding backward pass does not match!"
    print("Fast_RoPE_Embedding outputs and backward pass match!")

def validate_rope_embedding():
    test_slow_rope_embedding()
    test_fast_rope_embedding()
    print("All tests passed!")

if __name__ == "__main__":
    validate_rope_embedding()
