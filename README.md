### Analysis of Program Representations Based on Abstract Syntax Trees and Higher-Order Markov Chains for Source Code Classification Task

Code for the program classification algorithms described in the paper "Analysis of Program Representations Based on Abstract Syntax Trees and Higher-Order Markov Chains for Source Code Classification Task" [[1](https://doi.org/10.3390/fi15090314)].

### Getting Started

1. Install [Docker CE](https://docs.docker.com/engine/install/) and [GNU make](https://www.gnu.org/software/make/).
2. Clone the repository, then clone the submodules using `git submodule update --init --recursive`
3. Download the dataset [[2](https://doi.org/10.3390/data8060109)] from Zenodo and extract the `task-*.csv` files into `src/data`.
4. Classification targets can contain digits, so navigate to `external/code2vec/common.py` and apply the patch:
```diff
     @staticmethod
     def legal_method_names_checker(special_words, name):
-        return name != special_words.OOV and re.match(r'^[a-zA-Z|]+$', name)
+        return name != special_words.OOV
```
5. Run `make notebook` from repository root, run the notebooks.

### References

1. Gorchakov, A.V.; Demidova, L.A.; Sovietov, P.N. [Analysis of Program Representations Based on Abstract Syntax Trees and Higher-Order Markov Chains for Source Code Classification Task](https://doi.org/10.3390/fi15090314). Future Internet **2023**, 15, 314.
2. Demidova, L.A.; Andrianova, E.G.; Sovietov, P.N.; Gorchakov, A.V. [Dataset of Program Source Codes Solving Unique Programming Exercises Generated by Digital Teaching Assistant](https://doi.org/10.3390/data8060109). Data **2023**, 8 (6), p. 109.