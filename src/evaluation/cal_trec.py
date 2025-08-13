import argparse
import pytrec_eval
from pathlib import Path

def evaluate_run(run_file_path, evaluator, metrics):
    """
    Evaluates a single run file and prints the results.
    """
    # try:
    with open(run_file_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)
        
        results = evaluator.evaluate(run)
        # print(results)

        # 計算平均值
        avg_results = {}
        for metric in metrics:
            avg_results[metric] = sum(result[metric] for result in results.values()) / len(results)

        # 輸出結果
        for metric, value in avg_results.items():
            print(f'{metric}: {value:.4f}')
        print([round(value, 4) for metric, value in avg_results.items()])

    # except FileNotFoundError:
    #     print(f"Error: Run file not found at {run_file_path}")
    # except Exception as e:
    #     print(f"Error processing file {run_file_path.name}: {e}")
    # finally:
    #     print("-" * (20 + len(run_file_path.name)))
    #     print()

def main():
    """
    Main function to evaluate a single TREC run file.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a single TREC run file using pytrec_eval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to a single TREC run file."
    )
    parser.add_argument(
        "--qrels_file",
        default="data/qrels/10q_qrels.txt",
        help="Path to the QRELs file."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    qrels_file = Path(args.qrels_file)

    if not input_path.is_file():
        print(f"Error: Input path is not a file: {input_path}")
        return

    if not qrels_file.is_file():
        print(f"Error: QRELs file not found or is not a file: {qrels_file}")
        return

    try:
        with qrels_file.open('r') as f_qrel:
            qrels = pytrec_eval.parse_qrel(f_qrel)
    except Exception as e:
        print(f"Error parsing QRELs file {qrels_file}: {e}")
        return

    # Define the set of metrics to compute
    print(f"--- Evaluating: {input_path} ---")
    metrics = ['map_cut_10', 'map_cut_100', 'map_cut_1000', 'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_1000', 'recall_10', 'recall_100', 'recall_1000', 'P_10', 'P_100', 'P_1000']
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    evaluate_run(input_path, evaluator, metrics)

if __name__ == "__main__":
    main()
