import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import tensorflow as tf

class LimeExplainer:
    
    def __init__(self, model, test_ds, num_features=10, top_candidate=0, img_height=299, img_width=299):
        self.model = model
        self.test_ds = test_ds
        self.num_features = num_features
        self.top_candidate = top_candidate
        self.img_height = img_height
        self.img_width = img_width

    def predict_fn(self, images):
        # LIME sends a batch of images
        images = tf.image.resize(images, (self.img_height, self.img_width))
        preds = self.model.predict(images)
        return preds

    def explain_instance(self, image):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image,
            classifier_fn=self.predict_fn,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        return explanation

    def show_explanation(self, explanation):
        label = explanation.top_labels[self.top_candidate]
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=True,
            num_features=self.num_features,
            hide_rest=False
        )
        plt.figure(figsize=(6, 6))
        plt.title(f"LIME Explanation - Top Label {label}")
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.axis('off')
        plt.show()

    def heatmap(self, explanation):
        label = explanation.top_labels[self.top_candidate]
        dict_heatmap = dict(explanation.local_exp[label])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

        plt.figure(figsize=(6, 6))
        plt.title("LIME Feature Weights")
        plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
        plt.colorbar()
        plt.axis('off')
        plt.show()

        return heatmap
