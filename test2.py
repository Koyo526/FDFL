import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import csv
import scienceplots

# 利用可能なスタイルを確認
print(plt.style.available)

# ファイルパスの確認
file_path = '/home/murakata/FDFL/tmp2/clinet_token_history.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# データの読み込み
df = pd.read_csv(file_path)

# データフレームの列名を確認
print(df.columns)

# 各クライアントのトークンの総量の累積を計算
cumulative_tokens = df.iloc[:, 1:].cumsum()

# グラフのスタイルを設定
plt.style.use(['science', 'grid', 'no-latex'])

# カラーマップを設定
# colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns) - 1))
colors = ['blue', 'orange', 'green', 'red', 'purple']
# マーカーのリストを設定
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']

# グラフの描画
plt.figure(figsize=(8, 5))
for i, (color, marker) in enumerate(zip(colors, markers), start=1):
    plt.plot(df.iloc[:, 0], cumulative_tokens.iloc[:, i-1], label=f'Node{i}', marker=marker, color=color,lw=1.5, markersize=7)
# グリッドを追加
plt.grid(True, linestyle='--', alpha=0.7)

# タイトルとラベルを追加
# plt.title('Cumulative Client Token Over Rounds', fontsize=18)
plt.xlabel('Round', fontsize=14)
plt.ylabel('Cumulative Node Token', fontsize=14)
# x軸を整数に設定
plt.xticks(np.arange(df.iloc[:, 0].min(), df.iloc[:, 0].max() + 2, 2), fontsize=16)
plt.yticks(np.arange(0,45,10),fontsize=16)

# 凡例の位置を変更
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# グラフの表示
plt.tight_layout()
plt.show()

# グラフの保存
plt.savefig('tmp2/s4_node_token_amount_history.png', bbox_inches='tight')