#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
import re
from wordcloud import WordCloud


file = "exampleText.txt" # 需要统计词频的文件。
stop_words = "stop_words.txt" # 停用词文件。
font = "simhei.ttf" # 选用字体。可选：simfang.ttf 等。


if __name__ == '__main__': 
    with open(file, 'r', encoding='utf-8') as f:
        context = f.read()
    
    
    # <-------------------
    # 清理标点。
    # 最好先不清理标点，因为会影响jieba分词的断句。尽可能是在停用词"stop_words.txt"中添加停用标点。
    pattern = re.compile(r'[\u4e00-\u9fa5]') # 仅保留中文字符。可以重新修改为包含中英文字符。
    filterdata = re.findall(pattern, context)
    clean_context = "".join(filterdata)
    # --------------------->
    
    
    segment = jieba.lcut(clean_context) # 清理标点的文章
    # segment = jieba.lcut(context) # 不清理标点的文章
    words_df = pd.DataFrame({'segment': segment})
    
    # 清理停用词
    stop_words = pd.read_csv(stop_words, index_col = False, sep = "\t", names=['stopword'], encoding = 'utf-8')
    words_df = words_df[~words_df.segment.isin(stop_words.stopword)]
    
    # 统计词频
    words_stat = words_df.groupby(by=['segment'])['segment'].aggregate({"计数":np.size})
    words_stat = words_stat.reset_index().sort_values(by=["计数"], ascending = False)

    # 用词云进行显示
    wordcloud = WordCloud(font_path = font, width=600, height=600, max_words=200, background_color = "white", max_font_size = None, random_state = 50)
    word_frequence = {x[0]:x[1] for x in words_stat.head(200).values} # 出现最多的200条词条
    word_frequence_list = {}
    for key in word_frequence:
        word_frequence_list[key] = word_frequence[key]
    wordcloud = wordcloud.fit_words(word_frequence_list) # 根据出现频率生成词云
    wordcloud.to_file("wordcloud.jpg") # 保存为图片
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show() # 预览图片
