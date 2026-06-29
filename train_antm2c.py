"""Training entry point for the AntM2C (5-domain) benchmark.

Trains a mask-based multi-modal, multi-domain CTR model. The mask network is
warmed up first, then its temperature is annealed so that the soft modality
gates gradually become hard 0/1 selections.

Example:
    python train_antm2c.py --model dnn1 --data_dir /path/to/AntM2C --cuda 0
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn import metrics

from utils import train_utils
from modules import mask_models_antm2c
from dataloader.loader import Antm2cLoader

parser = argparse.ArgumentParser(description="AMS trainer (AntM2C)")
parser.add_argument("--dataset", type=str, default="Antm2c", help="dataset name")
parser.add_argument("--model", type=str, default="dnn1", help="fusion variant: dnn1-dnn6")

# Dataset information.
parser.add_argument("--data_dir", type=str, default="data/AntM2C", help="data directory with TFRecords / npy files")
parser.add_argument("--image_dim", type=int, default=512, help="image feature dimension")
parser.add_argument("--text_dim", type=int, default=768 * 6, help="text feature dimension")
parser.add_argument("--feat_num", type=int, default=152491, help="total number of id features")
parser.add_argument("--id_field_num", type=int, default=2, help="number of id fields (user, item)")

# Training hyper-parameters.
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--l2", type=float, default=3e-7, help="L2 regularization")
parser.add_argument("--bsize", type=int, default=512, help="batch size")
parser.add_argument("--optim", type=str, default="Adam", help="optimizer type")
parser.add_argument("--max_epoch", type=int, default=5, help="maximum epochs")
parser.add_argument("--save_dir", type=Path, default="save_antm2c", help="model checkpoint path")
parser.add_argument("--early_num", type=int, default=1, help="early stop patience")
parser.add_argument("--warmup_rate", type=float, default=0, help="fraction of first-epoch steps used to warm up the mask")
parser.add_argument("--batch_num", type=int, default=16673, help="approx. number of training batches per epoch (drives temperature annealing)")

# Neural network hyper-parameters.
parser.add_argument("--dim", type=int, default=128, help="embedding dimension")
parser.add_argument("--mlp_dims", type=int, nargs="+", default=[1024, 512, 256], help="mlp layer sizes")
parser.add_argument("--mlp_dropout", type=float, default=0.0, help="mlp dropout rate")
parser.add_argument("--mlp_bn", action="store_true", default=False, help="enable mlp batch normalization")
parser.add_argument("--cross", type=int, default=3, help="number of cross layers")
parser.add_argument("--projection_dim", type=int, default=64, help="shared projection dimension")

# Mask network hyper-parameters.
parser.add_argument("--m_lr", type=float, default=3e-5, help="mask network learning rate")
parser.add_argument("--final_temp", type=float, default=1000, help="final annealing temperature")
parser.add_argument("--thre", type=float, default=0.5, help="mask binarization threshold")
parser.add_argument("--lambda1", type=float, default=0.1, help="weight of the mask KL regularization")
parser.add_argument("--lambda2", type=float, default=0.55, help="reserved hyper-parameter")

# Device information.
parser.add_argument("--cuda", type=int, choices=range(-1, 8), default=1, help="cuda device id, -1 for cpu")

args = parser.parse_args()

MY_SEED = 2025
torch.manual_seed(MY_SEED)
torch.cuda.manual_seed_all(MY_SEED)
np.random.seed(MY_SEED)
random.seed(MY_SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["NUMEXPR_MAX_THREADS"] = "8"


class Trainer(object):
    def __init__(self, opt):
        self.lr = opt["lr"]
        self.l2 = opt["l2"]
        self.bs = opt["bsize"]
        self.model_dir = opt["save_dir"]
        self.early_num = opt["early_num"]
        self.dataloader = Antm2cLoader(opt["data_dir"])
        self.device = train_utils.getDevice(opt["cuda"])
        self.network = mask_models_antm2c.getModel(opt["model"], opt["model_opt"]).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.domain_list = [0, 1, 2, 3, 4]

        self.m_lr = opt["m_lr"]
        self.thre = opt["thre"]
        self.lambda1 = opt["lambda1"]
        self.lambda2 = opt["lambda2"]

        self.optim = mask_models_antm2c.getOptim(
            self.network, opt["optimizer"], self.lr, self.m_lr, self.l2
        )
        self.logger = train_utils.get_log(opt["model"])

        self.opt = opt
        self.kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        # Per-domain accumulators for the text/image selection statistics.
        self.mask = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.domain_len = [1, 1, 1, 1, 1]
        self.text_select_rate_accumulate_step = {f"domain_{i}": [] for i in self.domain_list}
        self.image_select_rate_accumulate_step = {f"domain_{i}": [] for i in self.domain_list}
        self.batch_num = opt["batch_num"]
        self.warm_step = self.batch_num * args.warmup_rate
        self._print_model_params_info()

    def _print_model_params_info(self):
        """Print the number of (trainable) parameters of the network."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print("\n" + "=" * 60)
        print("Model parameter statistics:")
        print(f"  Total parameters:        {total_params:,}")
        print(f"  Trainable parameters:    {trainable_params:,}")
        print(f"  Non-trainable parameters:{non_trainable_params:,}")
        print("=" * 60 + "\n")

    def train_on_batch(self, label, id_feat, text, image, domain, step):
        self.network.train()
        self.network.zero_grad()
        label, id_feat, text, image, domain = (
            label.to(self.device), id_feat.to(self.device), text.to(self.device),
            image.to(self.device), domain.to(self.device),
        )

        logit1, logit2, logit3, logit4, logit5, mask = self.network(id_feat, text, image, domain, step)
        logloss1 = self.criterion(logit1, label)
        logloss2 = self.criterion(logit2, label)
        logloss3 = self.criterion(logit3, label)
        logloss4 = self.criterion(logit4, label)
        logloss5 = self.criterion(logit5, label)

        target_one = torch.ones_like(mask)
        kl_loss = self.kl_loss(mask, target_one)

        # Each sample is supervised only by the head of its own domain.
        task_loss = torch.mean(
            logloss1 * torch.eq(domain, 0).type(torch.long)
            + logloss2 * torch.eq(domain, 1).type(torch.long)
            + logloss3 * torch.eq(domain, 2).type(torch.long)
            + logloss4 * torch.eq(domain, 3).type(torch.long)
            + logloss5 * torch.eq(domain, 4).type(torch.long)
        )

        if step < self.warm_step:
            mask = torch.zeros_like(mask).detach()
            loss = task_loss
        else:
            loss = task_loss + self.lambda1 * kl_loss
            mask = mask.long()
            for i in self.domain_list:
                dom_mask = torch.eq(domain, i).long().squeeze(1)
                self.mask[i][0] += torch.sum(mask[:, 0] * dom_mask).item()
                self.mask[i][1] += torch.sum(mask[:, 1] * dom_mask).item()
                self.domain_len[i] += dom_mask.sum().item()

        loss.backward()
        for optim in self.optim:
            optim.step()
        return loss.item()

    def eval_on_batch(self, id_feat, text, image, domain):
        self.network.eval()
        with torch.no_grad():
            id_feat, text, image, domain = (
                id_feat.to(self.device), text.to(self.device),
                image.to(self.device), domain.to(self.device),
            )
            logit1, logit2, logit3, logit4, logit5, mask = self.network(id_feat, text, image, domain, step=100000)
            logit = (
                logit1 * torch.eq(domain, 0).type(torch.long)
                + logit2 * torch.eq(domain, 1).type(torch.long)
                + logit3 * torch.eq(domain, 2).type(torch.long)
                + logit4 * torch.eq(domain, 3).type(torch.long)
                + logit5 * torch.eq(domain, 4).type(torch.long)
            )
            prob = torch.sigmoid(logit).detach().cpu().numpy()
        return prob

    def evaluate_val(self, on: str):
        preds, trues = [], []
        inference_times = []
        for id_feat, text, image, label, domain in self.dataloader.get_data(on, batch_size=self.bs):
            start_time = time.time()
            pred = self.eval_on_batch(id_feat, text, image, domain)
            inference_times.append(time.time() - start_time)
            label = label.detach().cpu().numpy()
            preds.append(pred)
            trues.append(label)
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            self.logger.info(
                f"Validation - average batch inference time: {avg_inference_time * 1000:.2f} ms "
                f"(over {len(inference_times)} batches)"
            )
        y_pred = np.concatenate(preds).astype("float64")
        y_true = np.concatenate(trues).astype("float64")
        auc = metrics.roc_auc_score(y_true, y_pred)
        loss = metrics.log_loss(y_true, y_pred)
        return auc, loss

    def evaluate_test(self, on: str):
        preds_dist = {d: [] for d in self.domain_list}
        trues_dist = {d: [] for d in self.domain_list}
        n_samples = {d: 0 for d in self.domain_list}

        for id_feat, text, image, label, domain in self.dataloader.get_data(on, batch_size=self.bs):
            pred = self.eval_on_batch(id_feat, text, image, domain)
            label = label.detach().cpu().numpy()
            domain = domain.detach().cpu().numpy()
            for d in self.domain_list:
                ind = np.nonzero(domain == d)[0]
                preds_dist[d].append(pred[ind])
                trues_dist[d].append(label[ind])
                n_samples[d] += len(ind)

        auc, loss = {}, {}
        for d in self.domain_list:
            y_pred = np.concatenate(preds_dist[d]).astype("float64")
            y_true = np.concatenate(trues_dist[d]).astype("float64")
            auc[d] = metrics.roc_auc_score(y_true, y_pred)
            loss[d] = metrics.log_loss(y_true, y_pred)
        total = sum(n_samples.values())
        auc["all"] = sum(auc[d] * n_samples[d] / total for d in self.domain_list)
        return auc, loss

    def train(self, epochs):
        best_auc = 0.0
        erly_num = self.early_num
        early_stop = False
        te_auc = None
        te_loss = None
        temp_increase = self.opt["final_temp"] ** (1.0 / (self.batch_num - 1))

        self.network.thre = self.thre
        self.network.warm_step = self.warm_step
        train_batch_times = []
        for epoch_idx in range(int(epochs)):
            train_loss = 0.0
            step = 0
            if epoch_idx > 0:
                self.network.warm_step = 0
                self.warm_step = 0
            for id_feat, text, image, label, domain in self.dataloader.get_data_shuffle("train", batch_size=self.bs):
                if step > self.warm_step or epoch_idx > 0:
                    self.network.temp *= temp_increase
                start_time = time.time()
                loss = self.train_on_batch(label, id_feat, text, image, domain, step)
                train_batch_times.append(time.time() - start_time)
                train_loss += loss
                step += 1
                if step % 1000 == 0:
                    self.logger.info(
                        "[Epoch {epoch:d} | Step :{setp:d} | Train Loss:{loss:.6f}".format(
                            epoch=epoch_idx, setp=step, loss=loss
                        )
                    )
                if step % 50 == 0:
                    for d in self.domain_list:
                        self.text_select_rate_accumulate_step[f"domain_{d}"].append(self.mask[d][0] / self.domain_len[d])
                        self.image_select_rate_accumulate_step[f"domain_{d}"].append(self.mask[d][1] / self.domain_len[d])

            if train_batch_times:
                avg_train_batch_time = np.mean(train_batch_times)
                self.logger.info(
                    f"[Epoch {epoch_idx}] average batch training time: {avg_train_batch_time * 1000:.2f} ms "
                    f"(over {len(train_batch_times)} batches)"
                )
            train_loss /= step
            val_auc, val_loss = self.evaluate_val("val")
            self.logger.info(
                "[Epoch {epoch:d} | Train Loss: {loss:.6f} | Val AUC: {val_auc:.6f}, Val Loss: {val_loss:.6f}]".format(
                    epoch=epoch_idx, loss=train_loss, val_auc=val_auc, val_loss=val_loss
                )
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.network.state_dict(), self.model_dir)
                erly_num = self.early_num
            else:
                erly_num -= 1
                if erly_num <= 0:
                    self.network.load_state_dict(torch.load(self.model_dir))
                    self.network.to(self.device)
                    early_stop = True
                    te_auc, te_loss = self.evaluate_test("test")
                    for d in self.domain_list:
                        self.logger.info(
                            "Early stop at epoch {epoch:d}|Test AUC{d:}: {te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".format(
                                epoch=epoch_idx, d=d, te_auc=te_auc[d], te_loss=te_loss[d]
                            )
                        )
                    print("All_auc:{au_auc:.6f}".format(au_auc=te_auc["all"]))
                    break

        if not early_stop:
            te_auc, te_loss = self.evaluate_test("test")
            for d in self.domain_list:
                self.logger.info(
                    "Final Test AUC{d:}: {te_auc:.6f}, Test Loss{d:}:{te_loss:.6f}".format(
                        d=d, te_auc=te_auc[d], te_loss=te_loss[d]
                    )
                )
            print("All_auc:{au_auc:.6f}".format(au_auc=te_auc["all"]))

        # Dump the per-domain modality selection rates over training steps.
        with open(f"antm2c_mask_log_{args.model}.txt", "a") as f:
            for d in self.domain_list:
                f.write(f"domain_{d}_text_rate:\n{self.text_select_rate_accumulate_step[f'domain_{d}']}\n")
                f.write(f"domain_{d}_image_rate:\n{self.image_select_rate_accumulate_step[f'domain_{d}']}\n")


def main():
    sys.path.extend(["./modules", "./dataloader", "./utils"])

    model_opt = {
        "latent_dim": args.dim, "feat_num": args.feat_num, "id_field_num": args.id_field_num,
        "mlp_dropout": args.mlp_dropout, "use_bn": args.mlp_bn, "mlp_dims": args.mlp_dims,
        "cross": args.cross, "image_dim": args.image_dim, "text_dim": args.text_dim,
        "projection_dim": args.projection_dim,
    }

    opt = {
        "model_opt": model_opt, "dataset": args.dataset, "model": args.model, "lr": args.lr, "l2": args.l2,
        "bsize": args.bsize, "epoch": args.max_epoch, "optimizer": args.optim, "data_dir": args.data_dir,
        "save_dir": args.save_dir, "cuda": args.cuda, "early_num": args.early_num, "lambda1": args.lambda1,
        "lambda2": args.lambda2, "final_temp": args.final_temp, "thre": args.thre, "m_lr": args.m_lr,
        "batch_num": args.batch_num,
    }
    print(opt)
    trainer = Trainer(opt)
    trainer.train(args.max_epoch)


if __name__ == "__main__":
    main()
