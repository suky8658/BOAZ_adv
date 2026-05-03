from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

DEFAULT_OLIST_PATHS = [
    str(DATA_DIR / "olist_orders_dataset.csv"),
    str(DATA_DIR / "olist_order_items_dataset.csv"),
    str(DATA_DIR / "olist_customers_dataset.csv"),
    str(DATA_DIR / "olist_order_payments_dataset.csv"),
    str(DATA_DIR / "olist_order_reviews_dataset.csv"),
    str(DATA_DIR / "olist_products_dataset.csv"),
    str(DATA_DIR / "olist_sellers_dataset.csv"),
    str(DATA_DIR / "olist_geolocation_dataset.csv"),
    str(DATA_DIR / "product_category_name_translation.csv"),
]

from .analysis_agent import build_app


app = build_app()


if __name__ == "__main__":
    while True:
        user_question = input("\n분석 질문 입력 (종료하려면 exit): ").strip()
        if user_question.lower() == "exit":
            break

        csv_paths = DEFAULT_OLIST_PATHS
        print("\n[Info]")
        print("Olist 기본 데이터셋을 자동으로 사용합니다.")
        for path in csv_paths:
            print(f"- {path}")

        result = app.invoke(
            {
                "user_question": user_question,
                "csv_paths": csv_paths,
                "dataset_context": "",
                "analysis_plan": {},
                "python_draft": {},
                "execution_result": {},
                "validation": {},
                "retry_count": 0,
                "max_retries": 2,
                "feedback": "",
                "error": "",
                "final_answer": "",
            }
        )

        print("\n" + "=" * 60)
        print("[Question]")
        print(user_question)

        print("\n[Dataset Context]")
        print(result["dataset_context"])

        print("\n[Analysis Plan]")
        print(result["analysis_plan"])

        print("\n[Python Code]")
        print(result["python_draft"].get("code", ""))

        print("\n[Validation]")
        print(result["validation"])

        print("\n[Final Answer]")
        print(result["final_answer"])
        print("=" * 60)
