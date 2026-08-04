import sys

from approach_1 import config

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m approach_1.main <audio_file>")
        sys.exit(1)

    print(f"STT Provider: {config.STT_PROVIDER}")
    print(f"Evaluator Provider: {config.EVAL_PROVIDER}")

    stt_runner = config.get_stt_runner()
    evaluator = config.get_evaluator()

    print("1. Transcribing audio...")
    candidate_text = stt_runner.transcribe(sys.argv[1])
    print(f"   Candidate Transcript: '{candidate_text}'")

    print("2. Evaluating against gold transcript...")
    gold_text = "This is a mock transcript of the audio content."
    report = evaluator.evaluate(gold_text, candidate_text)

    print(f"\nEvaluation Results:")
    print(f"   Score: {report.overall_score}")
    print(f"   Status: {report.status}")

if __name__ == "__main__":
    main()
