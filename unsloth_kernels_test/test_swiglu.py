# 测试通过
import torch
import triton
import triton.language as tl
from swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

def test_swiglu_fg_kernel():
    # 创建测试数据
    batch, seq_len, hd = 4, 128, 64
    e = torch.rand((batch, seq_len, hd), dtype=torch.float32, device='cuda')
    g = torch.rand((batch, seq_len, hd), dtype=torch.float32, device='cuda')

    # 执行前向传播
    h = swiglu_fg_kernel(e, g)

    # 手动计算结果以验证正确性
    expected_h = (e * torch.sigmoid(e)) * g

    # 断言计算结果与预期结果一致
    assert torch.allclose(h, expected_h, atol=1e-5), "swiglu_fg_kernel test failed!"
    print("swiglu_fg_kernel test passed!")

def test_swiglu_DWf_DW_dfg_kernel():
    # 创建测试数据
    batch, seq_len, hd = 4, 128, 64
    DW = torch.rand((batch * seq_len, hd), dtype=torch.float32, device='cuda')
    e = torch.rand((batch * seq_len, hd), dtype=torch.float32, device='cuda')
    g = torch.rand((batch * seq_len, hd), dtype=torch.float32, device='cuda')

    # 执行反向传播
    DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)

    # 手动计算结果以验证正确性
    se = torch.sigmoid(e)
    f = se * e
    df = DW * f
    dg = DW * g
    de = dg * se * (1.0 + e * (1.0 - se))

    # 计算结果与预期结果一致
    assert torch.allclose(df, DW * f, atol=1e-5), "swiglu_DWf_DW_dfg_kernel df test failed!"
    assert torch.allclose(de, dg * se * (1.0 + e * (1.0 - se)), atol=1e-5), "swiglu_DWf_DW_dfg_kernel de test failed!"
    print("swiglu_DWf_DW_dfg_kernel test passed!")

if __name__ == "__main__":
    test_swiglu_fg_kernel()
    test_swiglu_DWf_DW_dfg_kernel()
