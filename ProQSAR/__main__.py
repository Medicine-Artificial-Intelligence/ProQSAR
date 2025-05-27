import argparse
import logging
import json
import os
import pickle
import matplotlib
import pandas as pd
from ProQSAR.qsar import ProQSAR
from ProQSAR.Config.config import Config


def parse_kv_pairs(arg_list):
    """
    Parse a list of key=value strings into a dict.
    If the first item looks like JSON (starts with '{'), parse as JSON.
    """
    if not arg_list:
        return None
    first = arg_list[0]
    if first.strip().startswith("{"):
        return json.loads(" ".join(arg_list))
    result = {}
    for item in arg_list:
        key, eq, val = item.partition("=")
        if not eq:
            raise argparse.ArgumentTypeError(f"Expected key=value, got '{item}'")
        try:
            parsed = json.loads(val)
        except Exception:
            parsed = val
        result[key] = parsed
    return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ProQSAR: full pipeline (train & predict), train-only, or predict-only."
    )
    # Mutually exclusive: train vs predict-only
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model_path",
        type=str,
        help="Path to existing ProQSAR pickle file (for predict-only mode).",
    )
    group.add_argument(
        "--data_dev",
        type=str,
        help="Path to development data file (CSV/JSON/GZ) (for full-run mode).",
    )

    # prediction data is optional in full-run, required in predict-only
    parser.add_argument(
        "--data_pred",
        type=str,
        default=None,
        help="Path to prediction data file (CSV/JSON/GZ).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help="Conformal alpha(s), e.g. --alpha 0.05 0.1",
    )

    # Common args
    parser.add_argument(
        "--activity_col",
        type=str,
        default=None,
        help="Activity column name (required in full-run mode).",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default=None,
        help="ID column name (required in full-run mode).",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="SMILES",
        help="SMILES column name (default: SMILES).",
    )
    parser.add_argument(
        "--mol_col",
        type=str,
        default="mol",
        help="RDKit Mol column name (default: mol).",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="Project",
        help="Directory name under which all outputs will be saved.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--scoring_target",
        type=str,
        default=None,
        help="Primary metric for model selection (e.g., 'rmse', 'roc_auc').",
    )
    parser.add_argument(
        "--scoring_list",
        nargs="+",
        default=None,
        help="List of metrics to report (e.g., --scoring_list rmse mae).",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="CV folds",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="CV repeats",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logging.log",
        help="Log filename",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    # Config shortcuts & params
    for component in (
        "standardizer","featurizer","splitter","duplicate","missing",
        "lowvar","univ_outlier","kbin","multiv_outlier","rescaler",
        "feature_selector","model_dev","optimizer","ad","conf_pred"
    ):
        parser.add_argument(
            f"--{component}",
            metavar="KEY=VAL",
            nargs="+",
            type=str,
            default=None,
            help=(
                f"Parameters for Config.{component}; "
                "either a JSON object (`'{\"param\":value}'`) "
                "or space-separated key=value pairs"
            ),
        )
    return parser.parse_args()


def read_data(filepath: str) -> pd.DataFrame:
    """Load data from CSV or JSON (gzipped or plain)."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in {".gz", ".json", ".jsonl"}:
            return pd.read_json(filepath, lines=False)
        elif ext == ".csv":
            return pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        logging.error(f"Failed to load data from {filepath}: {e}")
        raise


def main():
    matplotlib.use("Agg")
    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # If predict-only, data_pred is required
    if args.model_path and not args.data_pred:
        raise ValueError("--data_pred is required when using --model_path")

    # build the Config(...) kwargs by parsing each component
    config_kwargs = {}
    for comp in (
        "standardizer","featurizer","splitter","duplicate","missing",
        "lowvar","univ_outlier","kbin","multiv_outlier","rescaler",
        "feature_selector","model_dev","optimizer","ad","conf_pred"
    ):
        config_kwargs[comp] = parse_kv_pairs(getattr(args, comp))

    config = Config(**config_kwargs)

    # Predict-only mode
    if args.model_path:
        logging.info("Loading model from %s", args.model_path)
        with open(args.model_path, "rb") as f:
            proqsar: ProQSAR = pickle.load(f)

        logging.info("Loading prediction data from %s", args.data_pred)
        data_pred = read_data(args.data_pred)

        logging.info("Running predict()")
        proqsar.predict(data_pred, alpha=args.alpha)
        
        return
    
    # Full-run mode
    logging.info("Loading development data from %s", args.data_dev)
    data_dev = read_data(args.data_dev)

    # prediction data is optional
    data_pred = None
    if args.data_pred:
        logging.info("Loading prediction data from %s", args.data_pred)
        data_pred = read_data(args.data_pred)

    try:
        proqsar = ProQSAR(
            activity_col=args.activity_col,
            id_col=args.id_col,
            smiles_col=args.smiles_col,
            mol_col=args.mol_col,
            project_name=args.project_name,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            scoring_target=args.scoring_target,
            scoring_list=args.scoring_list,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            log_file=args.log_file,
            log_level=args.log_level,
            config=config,
        )

        logging.info("Starting ProQSAR run_all()")
        # run_all will fit, test-predict, validate, analysis, summary, and
        # only predict(data_pred) if data_pred is not None
        proqsar.run_all(data_dev, data_pred, alpha=args.alpha)

        logging.info("ProQSAR pipeline completed successfully.")

    except Exception as e:
        logging.error("An error occurred during the ProQSAR run: %s", e, exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
