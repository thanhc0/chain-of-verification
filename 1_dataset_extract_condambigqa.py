from datasets import load_dataset
import pandas as pd
import sys


def load_ambigqa_dataset(sample_size: int = 20, output_csv: bool = True):
    """
    Tải subset của dataset AmbigQA, sửa lỗi Split Name.
    """

    DATASET_NAME = "ambig_qa"
    # Sửa lỗi ở đây: Sử dụng tên Split hợp lệ là 'validation'
    SPLIT_NAME = "validation"

    print(f"⏳ Đang thử tải dataset '{DATASET_NAME}' (split='{SPLIT_NAME}', sample={sample_size})...")

    try:
        # 1. Tải dataset
        dataset = load_dataset(DATASET_NAME)

        # 2. Kiểm tra và chọn mẫu
        if SPLIT_NAME not in dataset:
            # Nếu tên split vẫn sai (rất khó xảy ra sau khi sửa)
            print(f"❌ Lỗi: Split '{SPLIT_NAME}' không tìm thấy. Các split có sẵn: {list(dataset.keys())}")
            return None

        max_size = len(dataset[SPLIT_NAME])
        if sample_size > max_size:
            print(
                f"⚠️ Kích thước mẫu {sample_size} lớn hơn kích thước tối đa ({max_size}). Đang dùng kích thước tối đa.")
            sample_size = max_size

        subset = dataset[SPLIT_NAME].select(range(sample_size))

        # 3. Chuyển đổi sang DataFrame
        df = pd.DataFrame(subset)

        print(f"✅ Đã tải thành công {len(df)} mẫu từ {DATASET_NAME}.")

        # 4. Lưu ra CSV
        if output_csv:
            output_filename = f"{DATASET_NAME}_{sample_size}.csv"
            df.to_csv(output_filename, index=False, encoding="utf-8")
            print(f"💾 Đã lưu vào file: {output_filename}")

        # 5. Xem trước
        print("\n--- Xem trước mẫu (question, answers) ---")

        preview_cols = ["question", "answer"]
        existing_cols = [col for col in preview_cols if col in df.columns]

        if existing_cols:
            print(df.head(3)[existing_cols].to_markdown(index=False))
        else:
            print(df.head(3).to_markdown(index=False))

        return df

    except Exception as e:
        print(f"❌ Lỗi cuối cùng khi tải dataset AmbigQA: {e}")
        return None


# Ví dụ sử dụng
if __name__ == "__main__":
    # Thử nghiệm với tên Split đã sửa: 'validation'
    load_ambigqa_dataset(sample_size=100)