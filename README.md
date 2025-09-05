# Adversarially-Guided Cycle-Diffusion for Unpaired Underwater Image Enhancement

This repository contains the official PyTorch implementation for **AG-CycleDiffusion**, a novel hybrid generative model for high-quality, unsupervised underwater image enhancement. The model translates distorted underwater photographs into clear, terrestrial-style images without requiring paired training data.

The project's architecture is a synthesis of modern techniques from diffusion models and Generative Adversarial Networks (GANs), prioritizing final image quality and training stability.



## 📝 Project Overview

Underwater imagery often suffers from severe degradation, including color casts, low contrast, and backscatter, which hinders computer vision tasks. This project tackles the challenge of underwater image enhancement using an unpaired image-to-image translation approach. By leveraging a diffusion-based generative engine within a cycle-consistent adversarial framework, the model learns to restore natural colors and details, producing high-fidelity results.

---

## 🏛️ Core Architecture

The model is a hybrid system composed of four distinct neural networks, guided by a sophisticated multi-phase training curriculum.

* **Generative Engine**: The core translators are two **Conditional Denoising U-Nets** (`Underwater → Clear` and `Clear → Underwater`). These networks function as diffusion models, learning to generate a clean image by iteratively denoising from random noise, guided by a condition image from the opposite domain.
* **Adversarial Critics**: Two **PatchGAN Discriminators** are trained to distinguish real images from generated ones in each domain. Their feedback (the adversarial loss) pushes the U-Nets to create outputs that are sharp, detailed, and photorealistic.
* **Cycle-Consistency**: The model is trained with a cycle-consistency loss, ensuring that an image translated to the target domain and back (`Underwater → Clear → Underwater`) remains faithful to the original content. This is the key to learning from unpaired data.

---

## 🧪 Training Methodology

To ensure stability and high-quality convergence, the model is trained using an advanced curriculum broken into distinct phases.

### Key Features
* **Noise-Predicting U-Nets**: The core U-Nets are trained to predict the noise that should be removed from an image at any given timestep, which is a robust and standard approach for diffusion models.
* **Five Balanced Losses**: The training is driven by a carefully balanced combination of five loss functions:
    1.  **Adversarial Loss**: For style and realism.
    2.  **Cycle-Consistency Loss**: For content preservation.
    3.  **Perceptual Loss**: For sharpness and high-frequency details.
    4.  **Identity Loss**: For color stability and preventing unnecessary changes.
    5.  **Diffusion Reconstruction Loss**: To anchor the denoising process.
* **Three-Phase Curriculum**:
    * **Phase 1 (Denoise & Identity)**: Builds a stable foundation for the U-Nets.
    * **Phase 2 (Content Preservation)**: Activates Cycle and Perceptual losses to learn the structural mapping.
    * **Phase 3 (Adversarial Sharpening)**: Gradually introduces the GAN loss to achieve photorealism.

### Performance Optimizations
* **Fast Diffusion**: Uses a **Cosine Beta Schedule** with only **200 timesteps**.
* **Efficient Sampling**: Employs a **DDIM sampler** for fast inference (25-50 steps).
* **Mixed Precision (AMP)**: Leverages `torch.cuda.amp` for faster training and reduced memory footprint.
* **EMA Weights**: An Exponential Moving Average of the generator weights is maintained for more stable and higher-quality final outputs.

---

## 💿 Datasets

This model requires two large, unpaired datasets.

* **Domain A (Underwater Images)**: A large, diverse collection of real-world underwater images. This project aggregated **24,269 images** from several standard benchmarks:
    * [RUIE](https://github.com/dlut-dimt/Realworld-Underwater-Image-Enhancement-RUIE-Benchmark)
    * [EUVP](https://www.kaggle.com/datasets/pamuduranasinghe/euvp-dataset)
    * [UIEB](https://li-chongyi.github.io/proj_benchmark.html)
    * [SUIM](https://irvlab.cs.umn.edu/resources/suim-dataset)
* **Domain B (Clear Terrestrial Images)**: A high-resolution dataset of clear, natural images to serve as the target style. This project used a subset of **25,000 images** from:
    * [COCO 2017 Dataset (Train Images)](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

---

## 🚀 How to Use

### Training
1.  Organize your datasets into two folders (e.g., `./datasets/underwater` and `./datasets/land`).
2.  Update the paths and hyperparameters in the `Config` class at the top of `train.py`.
3.  Run the training script:
    ```bash
    python train.py
    ```
4.  Checkpoints and sample images will be saved to the directories specified in the `Config` class.

### Inference
1.  Update the paths in the `InferenceConfig` class at the top of `inference.py`.
    * `CHECKPOINT_PATH`: Point this to the `.pth` checkpoint file you want to use (the EMA weights are recommended, e.g., `EMA_U_w2l_step_200000.pth`).
    * `INPUT_IMAGE_PATH`: The path to the underwater image you want to enhance.
2.  Run the inference script:
    ```bash
    python inference.py
    ```
3.  The enhanced image will be saved to `enhanced_output.png`.

---

## 📚 References

This work is a synthesis of ideas from the following incredible research papers:
* *Denoising Diffusion Probabilistic Models* (Ho et al., 2020)
* *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (Zhu et al., 2017)
* *Tackling The Generative Learning Trilemma With Denoising Diffusion Gans* (Xiao et al., 2022)
* *Cycle-Dehaze: Enhanced CycleGAN for Single Image Dehazing* (Engin et al., 2018)
