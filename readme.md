## chatYuan 使用体验

* [github - chatYuan](https://github.com/clue-ai/ChatYuan)
* [ChatYuan-large-v2](https://huggingface.co/ClueAI/ChatYuan-large-v2/)

### 依赖安装

这两个是指定版本

```
pip3 install gradio==3.20.1
pip3 install transformers==4.26.1
```

Pytorch 安装实际情况进行安装

### 数据介绍

任务类型	
理解任务（acc，10类）		
分类 classify	
情感分析 emotion_analysis	
相似度计算 similar	
自然语言推理 nli	
指代消解 anaphora_resolution	
阅读理解 reading_comprehension	
关键词提取 keywords_extraction	
信息抽取 ner	
知识图谱问答 knowledge_graph	
中心词提取 Keyword_extraction	
生成任务（rouge，6类）		
翻译（英中、中英） nmt
摘要 summary
问答 qa	
生成（文章、问题生成）
改写 paraphrasemr
纠错 correct

