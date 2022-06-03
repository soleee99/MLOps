import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torchmetrics
from transformers import AutoModel


class ColaModel(pl.LightningModule):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", lr=1e-4):
        super(ColaModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2

        self.train_accuracy_metric = torchmetrics.Accuracy()
        self.val_accuracy_metric = torchmetrics.Accuracy()
        #self.f1_metric = torchmetrics.F1(num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(average="micro")
        self.recall_micro_metric = torchmetrics.Recall(average="micro")
        self.lr = lr
        #self.save_hyperparameters()


    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        h_cls = outputs.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        loss = F.cross_entropy(outputs, batch["label"])
        preds = torch.argmax(outputs, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["label"]
        )
        preds = torch.argmax(outputs, 1)
        loss = F.cross_entropy(outputs, batch["label"])

        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        #f1 = self.f1_metric(preds, labels)

        # Logging metrics
        #self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True)
        #self.log("valid/f1", f1, prog_bar=True)
        return {"labels": labels, "logits": outputs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
