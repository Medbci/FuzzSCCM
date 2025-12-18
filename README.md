# FuzzSCCM

This repository provides the official implementation of **Fuzzy symbolic convergent cross mapping: A causal coupling measure for EEG signals in disorders of consciousness patients**.

The paper associated with this code has been **accepted for publication**. If you use this code or any part of it in your research, please cite the corresponding paper.

---

## Citation

```bibtex
@article{Li2026FuzzSCCM,
  title   = {Fuzzy symbolic convergent cross mapping: A causal coupling measure for EEG signals in disorders of consciousness patients},
  author  = {Li, Tingting and An, Xingwei and Di, Yang and Wang, Honglin and Yan, Yujia and Liu, Shuang and Dong, Yueqing and Ming, Dong},
  journal = {Neural Networks},
  volume  = {195},
  pages   = {108318},
  year    = {2026},
  issn    = {0893-6080},
  doi     = {10.1016/j.neunet.2025.108318},
  url     = {https://www.sciencedirect.com/science/article/pii/S0893608025011992}
}
```

---

## Environment Versions

* **Python**: 3.12.6
* **MNE**: 1.6.1

---

## Project Structure

* `fuzzSCCM.py`
  Core implementation of the FuzzSCCM method.

  * Default parameters:

    * Embedding dimension: `E = 3`
    * Fuzzy parameter: `k = 0.5`
    * Time length: `time_length = 1000`
  * Usage notes:

    * Replace `your_path` with the directory containing your `.set` EEG file
    * Replace `your_out_path` with the desired output directory

---

## Notes

* This implementation is intended for research use.
* For reproducibility, please ensure that the environment versions listed above are used.

---

## License

Please cite the paper above when using this code in academic work.
