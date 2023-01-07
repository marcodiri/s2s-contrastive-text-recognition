# SeqCLR - Sequence-to-Sequence Contrastive Learning for Text Recognition
A framework for sequence-to-sequence contrastive learning (SeqCLR) of visual representations, applied to text recognition.

A PyTorch implementation of the paper by *Aberdam et al.* [1].

The implementation is partially based on [Clova AI's deep text recognition benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).

In the `kaggle_notebook` folder there are some example scripts I used for training:
- [supervised_baseline](https://github.com/marcodiri/s2s-contrastive-text-recognition/blob/master/kaggle_notebook/supervised-baseline/supervised-baseline.ipynb) is to train the raw encoder-decoder without encoder pre-training.
- For the pipeline described in the paper:
    1. [pre-train the encoder](https://github.com/marcodiri/s2s-contrastive-text-recognition/blob/master/kaggle_notebook/pre-training/pre-training.ipynb);
    2. [train the decoder](https://github.com/marcodiri/s2s-contrastive-text-recognition/blob/master/kaggle_notebook/decoder-train/decoder-train.ipynb) on top of the the encoder obtained in the previous step.

**N.B.**: This is a project I made for an university assignment (my first one in machine learning actually) and I cannot guarantee the correctness and is not in any way meant for any serious application as is. Please review the code before using it.

## References
[1] Aberdam, A., R. Litman, S. Tsiper, O. Anschel, R. Slossberg, S. Mazor, R. Manmatha, and P. Perona (2020).
Sequence-to-Sequence Contrastive Learning for Text Recognition. DOI: [10.48550/ARXIV.2012.10873](https://doi.org/10.48550/ARXIV.2012.10873).