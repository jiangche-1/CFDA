import os
import pandas as pd
import emoji
import re

def smart_emoji_augment(text, max_total_emoji=3):
    """
    1. 限制增强的 Emoji 描述总数 (max_total_emoji)
    2. 多个相同的 Emoji 只保留一个描述
    3. 彻底删除原文中的所有原始 Emoji 字符
    4. 不保留 (pos)/(neg) 等情感后缀，只留语义
    """
    # 提取所有 emoji
    emoji_list = emoji.emoji_list(text)
    if not emoji_list:
        return text

    seen_chars = set()
    to_replace = [] # 需要翻译成文字标签的项
    to_delete = []  # 需要直接移除的项

    for item in emoji_list:
        char = item['emoji']
        if char not in seen_chars and len(to_replace) < max_total_emoji:
            to_replace.append(item)
            seen_chars.add(char)
        else:
            to_delete.append(item)

    content_list = list(text)
    all_ops = []

    # 处理需要翻译的 Emoji
    for item in to_replace:
        char = item['emoji']
        # 仅获取语义描述并缩短
        desc = emoji.demojize(char).replace(":", "").replace("_", " ")
        short_desc = desc.replace("face with ", "").replace("face ", "")
        
        all_ops.append({
            'start': item['match_start'],
            'end': item['match_end'],
            'text': f" [{short_desc}] "
        })

    # 处理需要删除的重复/超量 Emoji
    for item in to_delete:
        all_ops.append({
            'start': item['match_start'],
            'end': item['match_end'],
            'text': "" 
        })

    # 从后往前执行替换，保护索引
    all_ops.sort(key=lambda x: x['start'], reverse=True)

    for op in all_ops:
        content_list[op['start']:op['end']] = list(op['text'])

    # 结果清理：确保 $T$ 独立并去除多余空格
    result = "".join(content_list).replace("$T$", " $T$ ")
    return " ".join(result.split())


def _read_tsv(path):
    df = pd.read_csv(path, sep="\t")
    n = df.shape[1]
    if n == 5:
        df.columns = ["index", "sentiment", "image_id", "tweet_content", "target"]
    elif n == 4:
        df.columns = ["index", "image_id", "tweet_content", "target"]
    else:
        raise ValueError(f"{path}: 读到 {n} 列，预期 4 或 5 列")
    return df.drop(columns=["index"])


def _read_test_labels(txt_path):
    lines = open(txt_path, encoding="utf-8").read().split("\n")
    labels, image_ids = [], []
    for i in range(0, len(lines) - 3, 4):
        labels.append(int(lines[i + 2].strip()) + 1)
        image_ids.append(lines[i + 3].strip())
    return labels, image_ids


def load_data(args, replace_emoji=True):
    base = os.path.join(args.data_dir, args.dataset)
    train_df = _read_tsv(os.path.join(base, "train.tsv"))
    val_df = _read_tsv(os.path.join(base, "dev.tsv"))
    test_df = _read_tsv(os.path.join(base, "test.tsv"))

    labels, image_ids = _read_test_labels(os.path.join(base, "test.txt"))
    if len(labels) != len(test_df):
        raise ValueError(f"test.txt 解析出 {len(labels)} 条，test.tsv 有 {len(test_df)} 行")
    if [str(x).strip() for x in test_df["image_id"]] != image_ids:
        raise ValueError("test.txt 与 test.tsv 顺序不一致，不能按行合并")
    test_df["sentiment"] = labels

    if replace_emoji:
        for df in (train_df, val_df, test_df):
            df["tweet_content"] = df["tweet_content"].apply(emoji.replace_emoji)

    return train_df, val_df, test_df
