# hpc-bahavioral-strategy

Code accompanying the manuscript. Each top-level folder is one model / analysis
approach:

| Folder | Language | Entry point |
|---|---|---|
| `MDP-HMM/` | MATLAB | `generate_x`, then `hdp_hmm_gibbs_crf_minibatch` |
| `RNN/` | Python | `run.ipynb` |
| `CSCG/` | Python | `intro.ipynb` |
| `TEM/` | Python (TensorFlow) | `run_tem.py` / `TEM_notebook.ipynb` |

## MDP-HMM (MATLAB)

1. Run **`generate_x`** first to produce a sequence of observations.
2. Use the generated sequence as the input to
   **`hdp_hmm_gibbs_crf_minibatch`** to train a hidden Markov model.

## RNN

Open and run `RNN/run.ipynb`.

## CSCG

Open and run `CSCG/intro.ipynb`.

## TEM (Tolman-Eichenbaum Machine)

- To look at the results, open `TEM/main/TEM_notebook.ipynb`.
- To run the model, run `python3 TEM/main/run_tem.py`.

The TEM implementation is based on the Tolman-Eichenbaum Machine; see
`TEM/main/README.md` for the original project details and dependencies.

## License

See `LICENSE`.
