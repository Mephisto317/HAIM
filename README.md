## Run

```bash
cd MKVCP/HAIM/code
g++ -std=c++11 -O3 HAIM.cpp -o haim-periodic
./haim-periodic ../data/dimacs/brock800_2.mis 698 1 1800
```

Arguments: `<instance> <k> <seed> <CPU time limit in seconds>`.

## Datasets

- [BHOSLIB](http://sites.nlsde.buaa.edu.cn/~kexu/benchmarks/graph-benchmarks.htm)
- [DIMACS](https://archive.dimacs.rutgers.edu/pub/challenge/graph/benchmarks/clique/)

Input graphs must be in text format; binary `.clq.b` files require conversion.
