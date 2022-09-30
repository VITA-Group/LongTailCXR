# Long-Tailed Classification of Thorax Diseases on Chest X-Ray: A New Benchmark Study

Gregory Holste, Song Wang, Ziyu Jiang, Thomas C. Shen, Ronald M. Summers, Yifan Peng, Zhangyang Wang<br>
**[Oral Presentation]** MICCAI Workshop on Data Augmentation, Labelling, and Imperfections (DALI). 2022.

[[Paper](https://link.springer.com/chapter/10.1007/978-3-031-17027-0_3)] | [[arXiv](https://arxiv.org/abs/2208.13365)] | [[Oral Presentation](https://drive.google.com/file/d/1IVylgwhPBs_HoaUQMvkX1R-7lXMANI7K/view?usp=sharing)]

## Abstract

<p align=center>
    <img src=figs/061322_log_mimic-lt_train.png height=600> <img src=figs/061322_log_nih-lt_train.png height=600>
</p>

Imaging exams, such as chest radiography, will yield a small set of common findings and a much larger set of uncommon findings. While a trained radiologist can learn the visual presentation of rare conditions by studying a few representative examples, teaching a machine to learn from such a “long-tailed” distribution is much more difficult, as standard methods would be easily biased toward the most frequent classes. In this paper, we present a comprehensive benchmark study of the long-tailed learning problem in the specific domain of thorax diseases on chest X-rays. We focus on learning from naturally distributed chest X-ray data, optimizing classification accuracy over not only the common “head" classes, but also the rare yet critical “tail” classes. To accomplish this, we introduce a challenging new long-tailed chest X-ray benchmark to facilitate research on developing long-tailed learning methods for medical image classification. The benchmark consists of two chest X-ray datasets for 19- and 20-way thorax disease classification, containing classes with as many as 53,000 and as few as 7 labeled training images. We evaluate both standard and state-of-the-art long-tailed learning methods on this new benchmark, analyzing which aspects of these methods are most beneficial for long-tailed medical image classification and summarizing insights for future algorithm design. The datasets, trained models, and code are available at https://github.com/VITA-Group/LongTailCXR.

-----

## Results & Model Weights

All trained model weights are available below. In the following table, best results are **bolded** and second-best results are <u>underlined</u>. See paper for full results (bAcc = [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)).

| Method | NIH-CXR-LT bAcc | MIMIC-CXR-LT bAcc | NIH-CXR-LT Weights | MIMIC-CXR-LT Weights |
| :--- | :---: | :---: | :---: | :---: |
| Softmax | 0.115 | 0.169 | [link](https://drive.google.com/file/d/1lzDBDwRRcKmYHaypyLc59MsbdSbOuq75/view?usp=sharing) | [link](https://drive.google.com/file/d/1iKMqNX_KvczyuJZAibmRSWU5cqKQ0psn/view?usp=sharing) |
| CB Softmax | 0.269 | 0.227 | [link](https://drive.google.com/file/d/1m0Xt_COF8SY5ZxKo3qrpwBDBjTnZIfd7/view?usp=sharing) | [link](https://drive.google.com/file/d/1GDCWZ0J1GhGdEqcEP9Ubp42p4tz_w58b/view?usp=sharing) |
| RW Softmax | 0.260 | 0.211 | [link](https://drive.google.com/file/d/1rvl_W3ZP6-059hevrP8FiRC3CM41DN65/view?usp=sharing) | [link](https://drive.google.com/file/d/1li4zP5-hCtazWVzC8Cp99o3hUU6t_PVv/view?usp=sharing) |
| Focal Loss | 0.122 | 0.172 | [link](https://drive.google.com/file/d/1YuMcxv9d8H1rH-nP-ccMmrT3MXJb0SGQ/view?usp=sharing) | [link](https://drive.google.com/file/d/1OxnUQxjAfsrydXcaJ6Xy2-WlA7jNNkRG/view?usp=sharing) |
| CB Focal Loss | 0.232 | 0.191 | [link](https://drive.google.com/file/d/1wOk9NlDrp4c52WjvJsqetxlEVfFJndBr/view?usp=sharing) | [link](https://drive.google.com/file/d/1ZzPJTA-OBLYphkzO5yF8r_VZgpoa6tXT/view?usp=sharing) |
| RW Focal Loss | 0.197 | 0.239 | [link](https://drive.google.com/file/d/1wMa6hd8J3jxlled7B66iDV43C3zdtL8l/view?usp=sharing) | [link](https://drive.google.com/file/d/1eTZ5K8HeDHzu3y_Nj0K2_MxPrK-E9MJg/view?usp=sharing) |
| LDAM | 0.178 | 0.165 | [link](https://drive.google.com/file/d/1i_kXKI4IXbWyABk6ChsqkSAaRu_LkmCi/view?usp=sharing) | [link](https://drive.google.com/file/d/1eT16iWKrpxJNIghLdaSq9Hr4aAt99CAL/view?usp=sharing) |
| CB LDAM | 0.235 | 0.225 | [link](https://drive.google.com/file/d/1p8uYrJH539Q9DgsEg7Ru_wOyRbZ1_taF/view?usp=sharing) | [link](https://drive.google.com/file/d/1mlOcyTuAN5SVlBXw-qyON7jXk7dHnhho/view?usp=sharing) |
| CB LDAM-DRW | 0.281 | 0.267 | [link](https://drive.google.com/file/d/17HMaldk6pwHEHZ-c3SJwPw3JWeYjCtI6/view?usp=sharing) | [link](https://drive.google.com/file/d/1YUtJq5iPgbd4O_p77EhhXJoA_CfvR8Ct/view?usp=sharing) |
| RW LDAM | 0.279 | 0.243 | [link](https://drive.google.com/file/d/1TZikaKB2sAqBA4o6bp9zVly463UAAftH/view?usp=sharing) | [link](https://drive.google.com/file/d/1X6p12_79o46OIBvSnnwERurv9x7eMf7t/view?usp=sharing) |
| RW LDAM-DRW | <u>0.289</u> | <u>0.275</u> | [link](https://drive.google.com/file/d/1hVe7y4sWE0o90UsZSRraQAU0UEmcu73c/view?usp=sharing) | [link](https://drive.google.com/file/d/1OVHRGfQVia3SU5UTBoQ2FtcRkiaYK63E/view?usp=sharing) |
| MixUp | 0.118 | 0.176 | [link](https://drive.google.com/file/d/1gP1LTgBQsrgCqzu3lyFK7TkcaPnSI-q7/view?usp=sharing) | [link](https://drive.google.com/file/d/1OjlkBsuumdvTtrUfBSGnCbONhXEk_cYf/view?usp=sharing) |
| Balanced-MixUp | 0.155 | 0.168 | [link](https://drive.google.com/file/d/1_GQXraEbGVMVu5WpAN8k1YB74M5yTNcV/view?usp=sharing) | [link](https://drive.google.com/file/d/16xA335kGktjH-O8iu8821LJc279bKjMh/view?usp=sharing) |
| Decoupling (cRT) | **0.294** | **0.296** | [link](https://drive.google.com/file/d/1nOqVEeZBmzyMM8fm46ziY6dqQHHcsAHm/view?usp=sharing) | [link](https://drive.google.com/file/d/1rbpyKxQsIGZbclMW0Fauxbt2TrxXoK8H/view?usp=sharing) |
| Decoupling (tau-norm) | 0.214 | 0.230 | -- | -- |

-----

## Data Access

Labels for the **MIMIC-CXR-LT** dataset presented in this paper can be found in the `labels/` directory. Labels for **NIH-CXR-LT** can be found at https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/174256157515. For both datasets, there is one csv file for each data split ("train", "balanced-val", "test", and "balanced-test").

-----

## Usage

To reproduce the results presented in this paper...
1. Register to download the MIMIC-CXR dataset from https://physionet.org/content/mimic-cxr/2.0.0/, and download the NIH ChestXRay14 dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC/.
2. Install prerequisite packages with Anaconda: `conda env create -f lt_cxr.yml` and `conda activate lt_cxr`.
3. Run all MIMIC-CXR-LT experiments: `bash run_mimic-cxr-lt_experiments.sh` (first changing the `--data_dir` argument to your MIMIC-CXR path).
4. Run all NIH-CXR-LT experiments: `bash run_nih-cxr-lt_experiments.sh` (first changing the `--data_dir` argument to your NIH ChestXRay14 path).

-----

## Citation

```
@inproceedings{holste2022long,
  title={Long-Tailed Classification of Thorax Diseases on Chest X-Ray: A New Benchmark Study},
  author={Holste, Gregory and Wang, Song and Jiang, Ziyu and Shen, Thomas C and Shih, George and Summers, Ronald M and Peng, Yifan and Wang, Zhangyang},
  booktitle={MICCAI Workshop on Data Augmentation, Labelling, and Imperfections},
  pages={22--32},
  year={2022},
  organization={Springer}
}
```

-----

## Contact

Feel free to contact me (Greg Holste) at gholste@utexas.edu with any questions!
