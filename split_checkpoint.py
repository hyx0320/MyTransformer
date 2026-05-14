import os
import torch

def split_checkpoint(checkpoint_path, output_dir, max_shard_size_mb=100):
    """
    将 checkpoint 中的模型权重按大小分片保存，
    并只保留 epoch / loss 等极简训练元信息。
    """
    # 加载原始 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model_state_dict"]

    os.makedirs(output_dir, exist_ok=True)

    max_size = max_shard_size_mb * 1024 * 1024  # 转换为字节
    shard_id = 1
    current_shard = {}
    current_size = 0

    for name, tensor in model_state.items():
        param_size = tensor.numel() * tensor.element_size()

        # 如果当前参数会超出限制且分片非空，先保存
        if current_size + param_size > max_size and current_shard:
            shard_path = os.path.join(output_dir, str(shard_id))
            torch.save(current_shard, shard_path)
            print(f"Saved shard {shard_id}: {len(current_shard)} params, {current_size / 1024**2:.2f} MB")
            shard_id += 1
            current_shard = {}
            current_size = 0

        current_shard[name] = tensor
        current_size += param_size

    # 保存最后一个分片
    if current_shard:
        shard_path = os.path.join(output_dir, str(shard_id))
        torch.save(current_shard, shard_path)
        print(f"Saved shard {shard_id}: {len(current_shard)} params, {current_size / 1024**2:.2f} MB")
        shard_id += 1

    # ---- 只保存必要的元信息，体积极小 ----
    meta = {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        # 如果需要其他标量/字符串信息可在此添加，但不要加 optimizer、scheduler 等大状态
    }
    torch.save(meta, os.path.join(output_dir, "training_state.pth"))
    print(f"Saved minimal training state (epoch={meta['epoch']}, loss={meta['loss']})")

    print(f"Total shards: {shard_id - 1}")

if __name__ == "__main__":
    src_checkpoint = "checkpoint/best_model.pth"
    dst_dir = "checkpoint/best_model"
    split_checkpoint(src_checkpoint, dst_dir, max_shard_size_mb=100)