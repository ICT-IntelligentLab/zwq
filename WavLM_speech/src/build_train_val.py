import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # 注意这里改成了 full
    input_csv = "data/iemocap_segments_full.csv"
    train_csv = "data/train.csv"
    val_csv = "data/val.csv"

    if not os.path.exists(input_csv):
        print(f"❌ 找不到文件 {input_csv}，请先运行 segment_prepare_iemocap.py")
        return

    df = pd.read_csv(input_csv)
    print(f"📄 全量数据: {len(df)} 条")

    # 随机拆分 80% / 20%
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print(f"✅ 训练集: {len(train_df)} 条")
    print(f"✅ 验证集: {len(val_df)} 条")

if __name__ == "__main__":
    main()