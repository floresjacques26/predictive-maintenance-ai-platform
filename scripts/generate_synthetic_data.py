"""Generate and persist synthetic sensor dataset.

Usage
-----
python scripts/generate_synthetic_data.py --n-machines 100 --output-dir data/synthetic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic_generator import SyntheticSensorDataGenerator
from src.utils.logger import get_logger

logger = get_logger("generate_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic sensor data.")
    parser.add_argument("--n-machines", type=int, default=100, help="Number of machines")
    parser.add_argument("--failure-horizon", type=int, default=30, help="Lookahead horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/synthetic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info(f"Generating dataset: {args.n_machines} machines, horizon={args.failure_horizon}")

    gen = SyntheticSensorDataGenerator(
        n_machines=args.n_machines,
        failure_horizon=args.failure_horizon,
        random_seed=args.seed,
    )
    df = gen.generate(output_dir=args.output_dir)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Class distribution:\n{df['failure_imminent'].value_counts(normalize=True).to_string()}")
    logger.info(f"Saved to {args.output_dir}/sensor_data.csv")


if __name__ == "__main__":
    main()
