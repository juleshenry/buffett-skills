import json
from pathlib import Path

from . import batch_pipeline


def main():
    ticker_symbol = "ZBH"
    batch_pipeline.logger.info(f"Starting batch pipeline for {ticker_symbol}...")
    final_output = batch_pipeline.analyze_company(ticker_symbol)

    print("\n--- FINAL BATCH PIPELINE RESULTS ---\n")
    print(json.dumps(final_output, indent=2))

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_filename = output_dir / f"{ticker_symbol}_analysis.json"
    with output_filename.open("w") as f:
        json.dump(final_output, f, indent=2)
    batch_pipeline.logger.info(f"Successfully saved analysis to {output_filename}")


if __name__ == "__main__":
    main()
