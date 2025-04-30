# GAN Performance Comparison: `nn.Linear` vs `NdLinear` on MNIST

This project explores and compares the performance of Generative Adversarial Networks (GANs) where the encoder is either based on PyTorch's standard `nn.Linear` or `NdLinear` (from https://github.com/ensemble-core/NdLinear). Our aim was to analyze differences in reconstruction quality, parameter efficiency, and latent space representation.

---

## 📊 Objective

To evaluate the impact of replacing traditional `nn.Linear` layers with `NdLinear` layers in the encoder and decoder part of a GAN model trained on the MNIST dataset.

We compared both models based on:

- **Structural Similarity Index (SSIM)**
- **Fréchet Inception Distance (FID)**
- **Test Loss (Reconstruction loss)**
- **Model Parameters Count**
- **Latent Space Visualization (t-SNE)**

---

## 🧠 Model Architecture

### Baseline GAN
- **Encoder:** Linear Autoencoder using `nn.Linear`
- **Decoder:** Linear layers

### Experimental GAN
- **Encoder:** NdLinear Autoencoder using `NdLinear`
- **Decoder:** NdLinear layers 

Both models share the same GAN training loop and optimization strategy.

---

## 📈 Results Summary

| Metric              | Linear Autoencoder | NdLinear Autoencoder |
|---------------------|--------------------|-----------------------|
| SSIM                | ✅ Higher           | ❌ Lower              |
| FID Score           | ✅ Better           | ❌ Worse              |
| Test Loss           | ✅ Lower            | ❌ Higher             |
| Parameter Count     | ❌ Higher           | ✅ Lower              |
| Latent t-SNE        | ✅ Distinct clusters| ❌ Overlapping        |

- Linear Autoencoder had better image quality and more distinct latent clusters, but at the cost of higher parameter count.
- NdLinear Autoencoder had fewer parameters, showing its potential for lightweight models, but underperformed in reconstruction quality and latent space separability.

---

## 🔍 Visualizations

### t-SNE of Latent Space
- **Linear AE:** Well-defined digit clusters
- **NdLinear AE:** Significant overlap among digits

<p align="center">
  <img src="assets/tsne_linear.png" width="300"/>
  <img src="assets/tsne_ndlinear.png" width="300"/>
</p>

---

## 🧪 Dataset

- **MNIST Digits** (60,000 training and 10,000 testing images)
- Preprocessed to be in the range [0, 1] and reshaped into flat vectors for input to linear layers

