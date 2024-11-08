# 数据集获取方式

方法一（推荐）

已将`datasets`文件夹上传至北航网盘`联邦学习小组资料/项目/2023-科委-大模型项目/资料分享/datasets.zip`中，只需要下载后覆盖掉原先`datasets`文件夹即可。

---

方法二

* ## [IMDB](https://huggingface.co/datasets/imdb)：

  huggingface链接：https://huggingface.co/datasets/imdb

  也可以通过运行`utils/data_getter.py`中的`get_imdb_dataset()`来获取。

  情感分类任务，二分类，0表示负面情感，1表示正面情感，语料来自互联网电影数据库（IMDB）的评论；

  > 🌰
  >
  > Label 0: Holy crap. This was the worst film I have seen in a long time. All the performances are fine, but there is no plot. Really! No plot! A bunch of clowns talk about this and that and that's your film. Ug... Robert Duvall's character...
  >
  > Label 1: A true classic. Beautifully filmed and acted. Reveals an area of Paris which is alive and filled with comedy and tragedy. Although the area of 'Hotel du Nord' and the Hotel itself still exists, it is not as gay (in the original sense of the...

* ## [GLUE](https://huggingface.co/datasets/glue)：

  huggingface链接：https://huggingface.co/datasets/glue

  也可以通过运行`utils/data_getter.py`中的`get_glue_dataset()`来获取。

  详情介绍链接：https://zhuanlan.zhihu.com/p/135283598

  * CoLA：单句分类任务，二分类，0表示不合乎语法，1表示合乎语法，语料来自语言理论的书籍和期刊；
  >🌰
  >
  >Label 0: Mary sent.
  >Label 1: She is proud.
  * SST-2：单句分类任务，二分类，0表示负面情感，1表示正面情感，语料来自电影评论中的句子和它们情感的人类注释；
  > 🌰
  > 
  > Positive: two central performances
  > Negative: monotone
  * MRPC：句子对相似性和释义任务，二分类任务，0表示负样本、不互为释义，1表示正样本、互为释义，语料来自在线新闻源自动抽取+人工注释；

    > 🌰
    >
    > Label 0: How do I solve 3^1/3? How do I solve (x^2-1)/(x-3)<0?
    >
    > Label 1: Why does the iPad Mini say not charging? Why is my iPad Mini not charging?

  * MNLI：自然语言推断任务，三分类，任务是预测前提语句是否包含假设（蕴含、矛盾、中立），语料来自通过众包方式对句子对进行文本蕴含标注的集合；

    > 🌰
    >
    > Entailment: How do you know? All this is their information again. This information belongs to them.
    >
    > Contradiction: Poirot, I exclaimed, with relief, and seizing him by both hands, I dragged him into the room. Poirot was now back and I was sorry that he would take over what I now considered my own investigation.
    >
    > Neutral: She smiled back. She was so happy she couldn't stop smiling.

* ## 其余数据集

  下载链接：https://github.com/moon-hotel/BertWithPretrained/tree/main/data
  
  或者参考各自文件夹下的`README.md`文件即可。

