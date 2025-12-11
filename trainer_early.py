#wandb 제거 + earlystop 추가 버전
#if execute this code, it still call wandb. need to change code
import os
import json
import csv
from math import inf
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm

from loss_meter import LossMeter
#from loss_boundary import make_boundary_weight_map

# ------------------------------
# Local file logger (CSV + JSON)
# ------------------------------
class LocalLogger:
    def __init__(self, config: Dict[str, Any]):
        wandb_name = (
            config.get("wandb", {}).get("name")
            or config.get("experiment_name")
            or "exp"
        )
        exp_root = config.get("exp_dir", "./runs")
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(exp_root) / f"{wandb_name}-{time_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        self.run_dir = run_dir
        self.config_path = run_dir / "config.json"
        self.metrics_path = run_dir / "metrics.csv"

        # config.json은 쓰기만 하면 됩니다 → "w"
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[LocalLogger] Failed to write config.json: {e}")

        # 새 러닝마다 metrics.csv를 새로 만듭니다 → "w"
        self._csv_file = open(self.metrics_path, "w", newline="", encoding="utf-8")
        self._csv_writer = None
        self._header_written = False

        print(f"[LocalLogger] Run directory: {self.run_dir.resolve()}")
        
    def log(self, data: Dict[str, Any], step: int = None, prefix: str = None):
        row = dict(data)
        if step is not None:
            row["step"] = int(step)
        if prefix:
            row["tag"] = str(prefix)

        # 1) 최초 호출: 관측된 키로 헤더 생성
        if not self._header_written:
            fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
            self._header_written = True
        else:
            # 2) 새로운 키가 생기면 헤더 확장
            missing = [k for k in row.keys() if k not in self._csv_writer.fieldnames]
            if missing:
                # (a) 현재까지의 내용을 디스크로 내보냄
                self._csv_file.flush()

                # (b) 같은 경로를 "r"로 별도 오픈해서 이전 행을 읽기
                with open(self.metrics_path, "r", newline="", encoding="utf-8") as rf:
                    old_rows = list(csv.DictReader(rf))

                # (c) 쓰기 핸들을 닫고, 확장된 헤더로 전체 재작성
                self._csv_file.close()
                new_fields = self._csv_writer.fieldnames + missing
                self._csv_file = open(self.metrics_path, "w", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=new_fields)
                self._csv_writer.writeheader()
                for r in old_rows:
                    self._csv_writer.writerow(r)

        # 3) 현재 행 추가
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def close(self):
        try:
            self._csv_file.close()
        except Exception:
            pass


class Trainer:
    def __init__(self, config=None, model=None, gen_set=None):
        self.gen_set = gen_set
        self.config = config or {}
        self.model = model

        self.val_count = 0
        self.train_count = 0
        self.step_count = 0

        # ---- W&B 제거/옵션화 ----
        self.use_wandb = bool(self.config.get("wandb", {}).get("wandb_on", False))
        if self.use_wandb:
            try:
                import wandb  # lazy import
                wandb.init(
                    entity=self.config["wandb"]["entity"],
                    project=self.config["wandb"]["project"],
                    notes=self.config["wandb"]["notes"],
                    tags=self.config["wandb"]["tags"],
                    name=self.config["wandb"]["name"],
                    config=self.config,
                )
                self._wandb = wandb
            except Exception as e:
                print(f"[W&B] init failed: {e}. Continue without W&B.")
                self.use_wandb = False
                self._wandb = None
        else:
            self._wandb = None

        # ---- Local logger (항상 사용) ----
        self.local_logger = LocalLogger(self.config)

        # ---- Early Stopping 설정 ----
        es_cfg = self.config.get("early_stop", {})
        self.max_epochs = int(self.config.get("max_epochs", 200))
        self.early_stop_patience = int(es_cfg.get("patience", 10))  # 0이면 비활성
        self.early_stop_min_delta = float(es_cfg.get("min_delta", 0.03))
        self.no_improve_epochs = 0

        self.best_val_loss = inf

    def train(self, epoch, data_loader):
        total_loss_meter = LossMeter()
        step_loss_meter = LossMeter()
        pre_step = self.step_count

        pbar = tqdm(
            enumerate(data_loader),
            total=len(data_loader) if hasattr(data_loader, "__len__") else None,
            desc=f"Train | epoch {epoch}",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch_item in pbar:
            loss = self.model.step(batch_idx, batch_item, "train")
            torch.cuda.empty_cache()

            total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
            step_dict = loss.get_loss_dict_for_print("step")
            step_loss_meter.aggr(step_dict)

            # tqdm 진행 정보
            show_key = "total_step" if "total_step" in step_dict else next(iter(step_dict.keys()))
            pbar.set_postfix({show_key: f"{step_dict[show_key]:.4f}"})

            # 스케줄러/스텝 로깅
            if ((batch_idx + 1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0) or \
               (self.step_count == pre_step and batch_idx == len(data_loader) - 1):

                log_dict = step_loss_meter.get_avg_results()
                try:
                    log_dict["step_lr"] = self.model.scheduler.get_last_lr()[0]
                except Exception:
                    pass

                self.local_logger.log(log_dict, step=self.step_count, prefix="step")

                if self.use_wandb:
                    try:
                        self._wandb.log(log_dict, step=self.step_count)
                    except Exception as e:
                        print(f"[W&B] log failed: {e}")

                self.step_count += 1
                try:
                    self.model.scheduler.step(self.step_count)
                except Exception:
                    pass

                step_loss_meter.init()

        # epoch 단위 train 평균
        avg_train = total_loss_meter.get_avg_results()
        self.local_logger.log(avg_train, step=self.step_count, prefix="train")
        self.train_count += 1

        if self.use_wandb:
            try:
                self._wandb.log(avg_train, step=self.step_count)
            except Exception as e:
                print(f"[W&B] log failed: {e}")

        self.model.save("train")

    def test(self, epoch, data_loader, save_best_model):
        total_loss_meter = LossMeter()

        pbar = tqdm(
            enumerate(data_loader),
            total=len(data_loader) if hasattr(data_loader, "__len__") else None,
            desc=f"Val   | epoch {epoch}",
            dynamic_ncols=True,
            leave=False,
        )

        for batch_idx, batch_item in pbar:
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))

        avg_total_loss = total_loss_meter.get_avg_results()
        current = avg_total_loss.get("total_val", None)

        # --- 개선 여부 계산 ---
        best = self.best_val_loss if self.best_val_loss != inf else current
        delta = best - current if current is not None else None
        improved = False

        if current is not None:
            improved = (self.best_val_loss == inf) or (current < self.best_val_loss - self.early_stop_min_delta)

        # --- 정보 출력 ---
        print(f"\n[Val Epoch {epoch}]")
        print(f"  current_val_loss     = {current:.6f}")
        print(f"  best_val_loss        = {self.best_val_loss:.6f}" if self.best_val_loss != inf else
            f"  best_val_loss        = (first epoch)")
        print(f"  loss_delta           = {delta:.6f}" if delta is not None else "  loss_delta           = None")
        print(f"  min_delta(required)  = {self.early_stop_min_delta}")
        print(f"  improved?            = {improved}")
        print(f"  no_improve_epochs    = {self.no_improve_epochs}")

        # --- 로그 저장 ---
        log_row = dict(avg_total_loss)
        log_row["early_best_val"] = float(self.best_val_loss if self.best_val_loss != inf else 1e30)
        log_row["early_no_improve_epochs"] = int(self.no_improve_epochs)
        log_row["loss_delta"] = float(delta if delta is not None else 0)
        self.local_logger.log(log_row, step=self.step_count, prefix="val")

        # W&B
        if self.use_wandb:
            try:
                self._wandb.log(avg_total_loss, step=self.step_count)
            except Exception as e:
                print(f"[W&B] log failed: {e}")

        # --- Early Stopping 갱신 로직 ---
        if improved:
            self.best_val_loss = current
            self.no_improve_epochs = 0
            print(f"  -> improvement detected! Best loss updated.")
            if save_best_model:
                self.model.save("val")
        else:
            self.no_improve_epochs += 1
            print(f"  -> no improvement ({self.no_improve_epochs}/{self.early_stop_patience})")

    def run(self):
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]

        for epoch in range(self.max_epochs):
            self.train(epoch, train_data_loader)
            self.test(epoch, val_data_loader, True)

            # 조기 종료 체크
            if self.early_stop_patience > 0 and self.no_improve_epochs >= self.early_stop_patience:
                print(f"[EarlyStop] No improvement for {self.no_improve_epochs} epochs "
                    f"(patience={self.early_stop_patience}). Stop training.")
                break 
