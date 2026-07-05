import json
from src.earnings_calls import load_cached_transcript_text
from src.management_governance import ManagementEvaluation

if __name__ == '__main__':
    print("\n--- Testing ManagementEvaluation (Fast/Truncated) ---")
    try:
        me = ManagementEvaluation()
        # Truncate transcript to first 10,000 characters to prevent Ollama from hanging
        transcript = load_cached_transcript_text("AAPL")[:10000]
        me_result = me.evaluate("AAPL", transcript=transcript)
        print(me_result)
    except Exception as e:
        print("Error:", e)

