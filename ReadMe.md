data文件夹 记录了词义消歧数据，由于商务印书馆的知识产权问题，我们仅公布了FiCLS-v2数据集中的测试集。
output文件夹 实验结果，包括主实验和消融实验1、2的结果。
train.py 词义生成模型训练相关代码

score_test.py 报告中Gloss Similarity Scoring、Gloss Similarity Scoring with Probability
、Gloss Similarity Scoring with MBRR三种义项评分方式的评测代码。

gsp_test.py 报告中Gloss Probability Scoring评分方式的评测代码。

utils.py 代码运行时使用的辅助函数相关代码。

