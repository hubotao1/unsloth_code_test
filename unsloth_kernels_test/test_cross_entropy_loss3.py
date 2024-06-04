# 测试通过
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from cross_entropy_loss import fast_cross_entropy_loss

#  PyTorch version 定义自定义自动微分函数的前向传播。它接收 logits（模型的原始预测）和 labels（真实标签）
class PyTorchCrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        # 计算 logits 的对数 softmax，F.log_softmax 将 logits 规范化为对数概率，沿着最后一个维度（dim=-1）进行
        log_probs = F.log_softmax(logits, dim=-1)
        # 计算对数概率和真实标签之间的负对数似然损失。reduction='none' 表示返回每个批次元素的损失，而不是求和或平均。
        losses = F.nll_loss(log_probs, labels, reduction='none')
        # 保存 logits、labels 和对数概率以供反向传播使用。ctx 是一个上下文对象，可以用来存储反向计算所需的信息
        ctx.save_for_backward(logits, labels, log_probs)
        return losses

    # 定义自定义自动微分函数的反向传播。它接收 dlosses，即损失对前向传播输出的梯度
    @staticmethod
    def backward(ctx, dlosses):
        logits, labels, log_probs = ctx.saved_tensors
        # 计算 logits 的梯度。torch.exp(log_probs) 将对数概率转换回概率。
        # dlosses.unsqueeze(1) 重塑 dlosses 以匹配广播所需的维度。两者的元素级乘法给出了梯度计算的初始部分
        grad_logits = torch.exp(log_probs) * dlosses.unsqueeze(1)
        # 调整真实标签索引的梯度。scatter_add_ 在 labels.unsqueeze(1) 指定的索引处添加指定值 (-dlosses.unsqueeze(1))。此操作确保真实标签的梯度得到正确调整
        grad_logits.scatter_add_(1, labels.unsqueeze(1), -dlosses.unsqueeze(1))
        # 返回 logits 的梯度。第二个 None 表示 labels 没有梯度，因为它们不是可学习参数
        return grad_logits, None

def compare_cross_entropy_loss():

    batch_size = 2
    seq_len = 3
    vocab_size = 10

    logits = torch.randn(batch_size, seq_len, vocab_size, device='cuda', requires_grad=True)
    # 创建一个整数张量 labels，形状为 (batch_size, seq_len)，在 0 到 vocab_size 之间随机取值
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    
    # PyTorch implementation
    pytorch_loss_fn = PyTorchCrossEntropyLoss.apply
    # 将 logits 变形为二维张量，形状为 (batch_size * seq_len, vocab_size)，labels 的形状是batch*seq_len 
    pytorch_loss = pytorch_loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    pytorch_loss_mean = pytorch_loss.mean()

    # Triton implementation
    triton_loss = fast_cross_entropy_loss(logits, labels)
    triton_loss_mean = triton_loss.mean()

    print(f"PyTorch Loss: {pytorch_loss_mean.item()}")
    print(f"Triton Loss: {triton_loss_mean.item()}")
    assert torch.allclose(pytorch_loss_mean,triton_loss_mean,atol=1e-5),"forward test failed!"
    print("forward test passed!")
    # Backward pass
    pytorch_loss_mean.backward()
    pytorch_grads = logits.grad.clone()
    logits.grad.zero_()

    triton_loss_mean.backward()
    triton_grads = logits.grad.clone()

    print(f"PyTorch Gradients: {pytorch_grads}")
    print(f"Triton Gradients: {triton_grads}")
    assert torch.allclose(pytorch_grads,triton_grads,atol=1e-5),"backward test failed!"
    print("backward test passed!")

compare_cross_entropy_loss()
