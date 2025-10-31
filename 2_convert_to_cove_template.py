import pandas as pd
import os


def prepare_and_ensure_cove_cols(input_file_path, output_file_path):
    """
    Tải file CSV, đổi tên 'best_answer' thành 'true_answer', và chỉ thêm
    các cột 'cove_answer' và 'score' nếu chúng chưa tồn tại.
    Giữ nguyên tất cả các cột và dữ liệu khác.
    """
    print(f"Bắt đầu kiểm tra và chuẩn bị file: {input_file_path}")

    # 1. Tải file CSV gốc
    try:
        # Thử tải với encoding mặc định (utf-8), nếu lỗi thì thử latin1
        df = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file tại đường dẫn: {input_file_path}")
        return

    print(f"✅ Đã tải thành công {len(df)} hàng từ file gốc.")

    df_columns = set(df.columns)
    modified = False

    # 2. Đổi tên cột 'best_answer' thành 'true_answer'
    if 'best_answer' in df_columns and 'true_answer' not in df_columns:
        df = df.rename(columns={'best_answer': 'true_answer'})
        df_columns.add('true_answer')  # Cập nhật tập hợp cột
        df_columns.remove('best_answer')
        modified = True
        print("🔄 Đã đổi tên cột 'best_answer' thành 'true_answer'.")
    elif 'true_answer' in df_columns:
        print("👍 Cột 'true_answer' đã tồn tại. Bỏ qua đổi tên.")
    else:
        print("⚠️ Cột 'best_answer' hoặc 'true_answer' không tồn tại.")

    # 3. Kiểm tra và thêm cột 'cove_answer'
    if 'cove_answer' not in df_columns:
        df['cove_answer'] = ""  # Khởi tạo chuỗi rỗng
        modified = True
        print("➕ Thiếu cột 'cove_answer'. Đã thêm (String rỗng).")

    # 4. Kiểm tra và thêm cột 'score'
    if 'score' not in df_columns:
        df['score'] = 0  # Khởi tạo giá trị 0
        modified = True
        print("➕ Thiếu cột 'score'. Đã thêm (Giá trị 0).")

    # 5. Lưu DataFrame đã cập nhật ra file CSV nếu có thay đổi
    if modified:
        df.to_csv(output_file_path, index=False, encoding='utf-8')
        print(f"🎉 Hoàn tất! File đã được cập nhật và lưu tại: {output_file_path}")
    else:
        print("👍 Không có thay đổi nào được thực hiện. File được giữ nguyên.")
        # Tùy chọn: nếu không thay đổi, bạn có thể bỏ qua việc lưu lại file


# =========================================================================
# 💡 VÍ DỤ CÁCH SỬ DỤNG
# =========================================================================

# Đặt đường dẫn file input và output của bạn
INPUT_CSV_FILE = "TruthfulQA_200.csv"  # Thay bằng tên file gốc của bạn
OUTPUT_CSV_FILE = "TruthfulQA_200_cove.csv"

# Chạy hàm xử lý
prepare_and_ensure_cove_cols(INPUT_CSV_FILE, OUTPUT_CSV_FILE)