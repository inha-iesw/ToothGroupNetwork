import os
import argparse
import torch
from glob import glob

from train_configs import train_config_maker
from runner import runner


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune TGNet with pretrained .h5")

    parser.add_argument("--model_name", required=True,
                        help="tgnet_fps | tgnet_bdl")
    parser.add_argument("--config_path", required=True,
                        help="train config file")
    parser.add_argument("--experiment_name", required=True,
                        help="experiment name")
    parser.add_argument("--input_data_dir_path", required=True,
                        help="folder containing *_sampled_points.npy")
    parser.add_argument("--pretrained_h5_path", required=True,
                        help="path to pretrained tgnet_fps.h5")

    return parser.parse_args()


def scan_npy_files(data_root):
    npy_files = sorted(glob(os.path.join(data_root, "*_sampled_points.npy")))
    if len(npy_files) == 0:
        raise RuntimeError("No *_sampled_points.npy found in dataset folder.")
    return npy_files


def main():
    args = parse_args()

    # ---------------------------------------------------------
    # 1) Config 생성 (txt 사용 없이)
    # ---------------------------------------------------------
    config = train_config_maker.get_train_config(
        args.config_path,
        args.experiment_name,
        args.input_data_dir_path,
        "",  # dummy
        ""   # dummy
    )

    # ---------------------------------------------------------
    # 2) npy 파일 스캔 + 8:2 split
    # ---------------------------------------------------------
    npy_files = scan_npy_files(args.input_data_dir_path)
    print(f"[INFO] Found {len(npy_files)} samples")

    split_idx = int(len(npy_files) * 0.8)
    train_list = npy_files[:split_idx]
    val_list = npy_files[split_idx:]

    print(f"[INFO] Train samples: {len(train_list)}")
    print(f"[INFO] Val samples:   {len(val_list)}")

    # Training generator가 txt 파일을 요구하므로 txt를 임시 생성한다.
    train_txt = os.path.join(args.input_data_dir_path, "train_auto.txt")
    val_txt   = os.path.join(args.input_data_dir_path, "val_auto.txt")

    with open(train_txt, "w") as f:
        for p in train_list:
            basename = os.path.basename(p).replace("_sampled_points.npy", "")
            f.write(basename + "\n")

    with open(val_txt, "w") as f:
        for p in val_list:
            basename = os.path.basename(p).replace("_sampled_points.npy", "")
            f.write(basename + "\n")

    config["train_data_split_txt_path"] = train_txt
    config["val_data_split_txt_path"] = val_txt

    # ---------------------------------------------------------
    # 3) PyTorch TGNet 모델 생성 (기존 start_train과 동일)
    # ---------------------------------------------------------
    if args.model_name == "tgnet_fps":
        from models.fps_grouping_network_model import FpsGroupingNetworkModel
        from models.modules.grouping_network_module import GroupingNetworkModule
        model = FpsGroupingNetworkModel(config, GroupingNetworkModule)

    elif args.model_name == "tgnet_bdl":
        from models.bdl_grouping_network_model import BdlGroupingNetworkModel
        from models.modules.grouping_network_module import GroupingNetworkModule
        model = BdlGroupingNetworkModel(config, GroupingNetworkModule)

    else:
        raise ValueError("Unsupported model_name.")

    # ---------------------------------------------------------
    # 4) Pretrained weight 로딩 (PyTorch state_dict)
    # ---------------------------------------------------------
    print(f"[INFO] Loading pretrained weights: {args.pretrained_h5_path}")

    state_dict = torch.load(args.pretrained_h5_path, map_location="cpu")
    model.module.load_state_dict(state_dict, strict=True)

    print("[INFO] Pretrained weights loaded successfully.")

    # ---------------------------------------------------------
    # 5) Runner 실행 → Trainer가 fine-tuning 수행
    # ---------------------------------------------------------
    print("[INFO] Start Fine-tuning...")
    runner(config, model)


if __name__ == "__main__":
    main()
