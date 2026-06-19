import pandas as pd
import random
from datetime import datetime, timedelta

# 1. テスト用の要素を定義（営業現場でよくある項目）
staff_list = ['鈴木 俊郎', '佐藤 美咲', '田中 健太', '高橋 くるみ', '渡辺 直樹']
actions = ['顧客情報閲覧', '商談議事録登録', 'メール送信', '見積書作成', '案件ステータス更新']
companies = ['株式会社A商事', '有限会社Bテクノロジー', 'C流通システム', 'Dフーズ', 'Eマニュファクチャリング']

# 2. 100件の架空ログデータを生成
data = []
start_date = datetime(2026, 4, 1) # 2026年4月からのデータ

for i in range(100):
    # ランダムな日時を生成（9:00〜18:00の間）
    random_days = random.randint(0, 60)
    random_hours = random.randint(9, 18)
    random_minutes = random.randint(0, 59)
    log_date = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
    
    data.append({
        '日時': log_date.strftime('%Y-%m-%d %H:%M'),
        '担当者': random.choice(staff_list),
        'アクション': random.choice(actions),
        '取引先企業': random.choice(companies),
        '滞在時間(分)': random.randint(1, 45) # 1分〜45分のランダムな滞在
    })

# 3. DataFrameに変換してCSVとして保存
df = pd.DataFrame(data)
# 昇順で並び替え
df = df.sort_values(by='日時')

# ExcelやLooker Studioで文字化けしないように「utf-8-sig」で保存
df.to_csv('sales_tool_log.csv', index=False, encoding='utf-8-sig')

print("サンプルデータ 'sales_tool_log.csv' が正常に作成されました！")