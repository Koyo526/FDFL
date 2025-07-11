
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime
import numpy as np
import scienceplots 


def plot_client_token_history(current_time: str, file_path: str):
    # 利用可能なスタイルを確認
    print(plt.style.available)


    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # データの読み込み
    df = pd.read_csv(file_path)

    # データフレームの列名を確認
    print(df.columns)

    # 各クライアントのトークンの総量の累積を計算
    cumulative_tokens = df.iloc[:, 1:].cumsum()

    # グラフのスタイルを設定
    plt.style.use(['science', 'grid', 'no-latex'])  # scienceplotsが利用できない場合
    # plt.style.use('seaborn-v0_8-whitegrid')  # 利用可能なスタイルを使用

    # カラーマップを設定
    # colors = plt.cm.tab10(np.linspace(0, 1, len(df.columns) - 1))
    # colors = ['blue', 'orange', 'green', 'red', 'purple']
    # Use 20 distinct colors 
    # colors = plt.cm.get_cmap('tab20', len(df.columns) - 1).colors
    colors = [
    'red',       # 0
    'blue',      # 1
    'green',     # 2
    'orange',    # 3
    'purple',    # 4
    'cyan',      # 5
    'magenta',   # 6
    'gold',      # 7   ※ yellow より見やすい
    'brown',     # 8
    'black',     # 9
    'pink',      # 10
    'olive',     # 11
    'teal',      # 12
    'navy',      # 13
    'lime'       # 14
]
    # マーカーのリストを設定
    # markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    # Use 15 distinct markers
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', '+', 'x', '|', '_']

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
    plt.yticks(np.arange(0,20,5),fontsize=16)

    # 凡例の位置を変更
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    # グラフの表示
    plt.tight_layout()
    plt.show()

    # グラフの保存
    save_dir = f'00_ClientTokens/{current_time}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/0_client_token_amount_history.png', bbox_inches='tight')
    # save csv same directory
    df.to_csv(f'{save_dir}/0_client_token_amount_history.csv', index=False)

if __name__ == "__main__":
    # download csv path 
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = f'tmp/client_token_history.csv'
    plot_client_token_history(current_time, file_path)
    print(f"Plot saved for {current_time}")
    print("Plotting completed successfully.")
