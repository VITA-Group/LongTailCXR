# LongTailCXR

## **[WORK IN PROGRESS]**

Code repository for **"Long-Tailed Classification of Thorax Diseases on Chest X-Ray: A New Benchmark Study"** by Gregory Holste, Song Wang, Ziyu Jiang, Tommy C. Shen, Ronald D. Summers, Yifan Peng, and Zhangyang Wang. To be presented at [DALI 2022](https://dali-miccai.github.io/), a MICCAI workshop.

-----

## Abstract

<p align=center>
    <img src=figs/061322_log_mimic-lt_train.png height=600> <img src=figs/061322_log_nih-lt_train.png height=600>
</p>

Imaging exams, such as chest radiography, will yield a small set of common findings and a much larger set of uncommon findings. While a trained radiologist can learn the visual presentation of rare conditions by studying a few representative examples, teaching a machine to learn from such a “long-tailed” distribution is much more difficult, as standard methods would be easily biased toward the most frequent classes. In this paper, we present a comprehensive benchmark study of the long-tailed learning problem in the specific domain of thorax diseases on chest X-rays. We focus on learning from naturally distributed chest X-ray data, optimizing classification accuracy over not only the common “head" classes, but also the rare yet critical “tail” classes. To accomplish this, we introduce a challenging new long-tailed chest X-ray benchmark to facilitate research on developing long-tailed learning methods for medical image classification. The benchmark consists of two chest X-ray datasets for 19- and 20-way thorax disease classification, containing classes with as many as 53,000 and as few as 7 labeled training images. We evaluate both standard and state-of-the-art long-tailed learning methods on this new benchmark, analyzing which aspects of these methods are most beneficial for long-tailed medical image classification and summarizing insights for future algorithm design. The datasets, trained models, and code are available at https://github.com/VITA-Group/LongTailCXR.

-----

## Results & Model Weights

All trained model weights are available below. In the following table, best results are **bolded** and second-best results are <u>underlined</u>. See paper for full results (bAcc = [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)).

| Method | NIH-LT bAcc | MIMIC-CXR-LT bAcc | NIH-LT Weights | MIMIC-CXR-LT Weights |
| :--- | :---: | :---: | :---: | :---: |
| Softmax | 0.115 | 0.169 | link | link |
| CB Softmax | 0.269 | 0.227 | link | link |
| RW Softmax | 0.260 | 0.211 | link | link |
| Focal Loss | 0.122 | 0.172 | link | link |
| CB Focal Loss | 0.232 | 0.191 | link | link |
| RW Focal Loss | 0.197 | 0.239 | link | link |
| LDAM | 0.178 | 0.165 | link | link |
| CB LDAM | 0.235 | 0.225 | link | link |
| CB LDAM-DRW | 0.281 | 0.267 | link | link |
| RW LDAM | 0.279 | 0.243 | link | link |
| RW LDAM-DRW | <u>0.289</u> | <u>0.275</u> | link | link |
| MixUp | 0.118 | 0.176 | link | link |
| Balanced-MixUp | 0.155 | 0.168 | link | link |
| Decoupling (cRT) | **0.294** | **0.296** | link | link |
| Decoupling ($\tau$-norm) | 0.214 | 0.230 | link | link |

-----

## Usage

