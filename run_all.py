"""Run the whole pipeline for one dataset with a single command.

  python run_all.py                          # iSarcasm: data, EDA, baseline, BiLSTM, comparison
  python run_all.py --transformer            # ... plus transformer fine-tuning (use a GPU)
  python run_all.py --dataset headlines --transformer
"""
import argparse
import subprocess
import sys

from src.data import DATASETS


def step(module, *args):
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n{'=' * 70}\n>>> {' '.join(cmd[2:])}\n{'=' * 70}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="isarcasm")
    parser.add_argument("--transformer", action="store_true", help="Also fine-tune the transformer")
    parser.add_argument("--use-context", action="store_true", help="Transformer uses the parent comment (Reddit)")
    args = parser.parse_args()
    ds = ["--dataset", args.dataset]

    step("src.data", *ds)
    step("src.eda", *ds)
    step("src.train_baseline", *ds)
    step("src.train_lstm", *ds)
    if args.transformer:
        step("src.train_transformer", *ds, *(["--use-context"] if args.use_context else []))
    step("src.compare", *ds)
    print("\nDone. Start the demo with:  streamlit run app.py")


if __name__ == "__main__":
    main()
