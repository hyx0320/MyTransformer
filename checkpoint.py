import torch
import os

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path="checkpoints/best_model.pth"):
    """保存断点：模型 + 优化器 + 训练状态"""
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    torch.save(checkpoint, save_path)
    print(f"[OK] 断点已保存：{save_path}")

def load_checkpoint_for_train(model, optimizer, load_path="checkpoints/best_model.pth", device="cuda"):
    """加载断点：用于继续训练"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    val_loss = checkpoint["val_loss"]
    print(f"[OK] 断点加载成功 | Epoch: {start_epoch} | Val Loss: {val_loss:.4f}")
    return model, optimizer, start_epoch, train_loss, val_loss

def load_model_for_inference(model, load_path="checkpoints/best_model.pth", device="cuda"):
    """仅加载模型权重：用于推理/翻译"""
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[OK] 推理模型加载完成！")
    return model