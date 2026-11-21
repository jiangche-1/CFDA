import pandas as pd
import emoji
from tqdm import tqdm

# 1. 加载数据
df_original = pd.read_csv('/root/userfolder/code/20241104image2text_conversion/image2text_conversion-main_1/dataset/twitter2017/test.tsv', sep='\t')
emoji_df = pd.read_csv('/root/userfolder_new/agent_model/tradition_model/image2text_conversion-main/utils/Emoji_Sentiment_Data_v1.0.csv')

# 模拟构建映射字典 (逻辑同前)
emoji_sentiment_map = {}
for _, row in emoji_df.iterrows():
    e = row['Emoji']
    scores = {'negative': row['Negative'], 'neutral': row['Neutral'], 'positive': row['Positive']}
    emoji_sentiment_map[e] = max(scores, key=scores.get)

# 2. 增强逻辑：带判断的转换
def get_emoji_sentiment_text(text):
    if not isinstance(text, str):
        return text, False
    
    emoji_data = emoji.emoji_list(text)
    if not emoji_data:
        return text, False # 返回 False 表示没有变化
    
    content_list = list(text)
    for item in sorted(emoji_data, key=lambda x: x['match_start'], reverse=True):
        char = item['emoji']
        desc = emoji.demojize(char).replace(":", "").replace("_", " ")
        sentiment = emoji_sentiment_map.get(char, "neutral")
        replacement = f" [{desc} ({sentiment})] "
        content_list[item['match_start']:item['match_end']] = list(replacement)
    
    return "".join(content_list).strip(), True # 返回 True 表示已增强

# 3. 串联数据集
new_rows = []
print("开始扫描并增强含 Emoji 的数据...")

for _, row in tqdm(df_original.iterrows(), total=len(df_original)):
    # 获取处理后的文本和是否改变的标记
    augmented_text, is_changed = get_emoji_sentiment_text(row['#3 String'])
    
    if is_changed:
        # 只有真正改变了（含有Emoji），才创建新行
        new_row = row.copy()
        new_row['#3 String'] = augmented_text
        new_row['index'] = f"{row['index']}_emoji_aug"
        new_rows.append(new_row)

# 将新增的行转化为 DataFrame
df_augmented_only = pd.DataFrame(new_rows)

# 纵向合并：原数据集 + 真正被增强的数据集
final_df = pd.concat([df_original, df_augmented_only], axis=0, ignore_index=True)

# 4. 保存
final_df.to_csv('merged_smart_augmented.csv', sep='\t', index=False)

print(f"\n--- 处理报告 ---")
print(f"原始样本总数: {len(df_original)}")
print(f"含 Emoji 并成功增强的样本数: {len(df_augmented_only)}")
print(f"最终合并后的总样本数: {len(final_df)}")