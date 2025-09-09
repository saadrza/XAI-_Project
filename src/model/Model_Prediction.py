import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf

class Model_Prediction:
    def __init__(self, model, test_ds, img_height=299, img_width=299):
        self.model = model
        self.test_ds = test_ds
        self.img_height = img_height
        self.img_width = img_width

    def predictions(self):
        y_true = []
        for _, label in self.test_ds.unbatch():
            y_true.append(int(label.numpy()))
        y_true = np.array(y_true)

        # Predict probabilities
        preds = self.model.predict(self.test_ds)
        y_pred_probs = tf.sigmoid(preds).numpy().squeeze()
        y_pred = (y_pred_probs >= 0.5).astype(int)

        return y_true, y_pred_probs, y_pred

    def classification_metrics(self, y_true, y_pred_probs, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs)
        cm = confusion_matrix(y_true, y_pred)

        print("Evaluation Metrics")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"AUC:       {auc:.4f}")

        # Plot confusion matrix
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        return acc, prec, rec, f1, auc, cm
