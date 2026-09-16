# hpc-behavioral-strategy

Code accompanying the manuscript. Each top-level folder is one model
approach:

| Folder | Language | Entry point |
|---|---|---|
| `HDP-HMM/` | MATLAB | `generate_x`, then `hdp_hmm_gibbs_crf_minibatch` |
| `RNN/` | Python | `run.ipynb` |
| `CSCG/` | Python | `intro.ipynb` |
| `TEM/` | Python | `run_tem.py` / `TEM_notebook.ipynb` |

## HDP-HMM (MATLAB)

1. Run `generate_x.m` first to produce a sequence of observations.
2. Use the generated sequence as the input to
   `hdp_hmm_gibbs_crf_minibatch.m` to train a hidden Markov model.

## RNN

Open and run `RNN/run.ipynb`.

## CSCG

Open and run `CSCG/intro.ipynb`.

## TEM (Tolman-Eichenbaum Machine)

Run `python3 TEM/main/run_tem.py`.
To look at the result, open and run `TEM/main/TEM_notebook.ipynb`.

## License

See `LICENSE`.
