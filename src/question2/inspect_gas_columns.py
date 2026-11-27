
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

def sort_key(p):
    nums = re.findall(r"\d+", p.name)
    return int(nums[0]) if nums else 999

files = sorted(RAW_DIR.glob("Attachment*.xlsb"), key=sort_key)

for file in files:
    df = pd.read_excel(file, engine="calamine")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    print("\n" + "=" * 75)
    print(file.name)
    print("数据维度：", df.shape)

    print("\n最后六列的字段名称：")
    print(list(df.columns[-6:]))

    print("\n最后三列的统计信息：")

    for col in df.columns[-3:]:
        s = pd.to_numeric(df[col], errors="coerce")

        print(f"\n字段：{col}")
        print("有效数量：", s.notna().sum())
        print("非零数量：", (s != 0).sum())
        print("不同数值数量：", s.nunique())
        print("最小值：", s.min())
        print("最大值：", s.max())
        print("前10个值：", s.head(10).tolist())
