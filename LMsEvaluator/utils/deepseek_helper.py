from openai import OpenAI
import json
import os, sys
from markdown_pdf import MarkdownPdf, Section
# from md2pdf.core import md2pdf
lmsDir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(lmsDir))
from web_databse.sql_manager import get_attack_record

def chatForTest(str):
  key = 'sk-919b0b8f52c24d43808aabadf2d30658'
  client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1')
  response = client.chat.completions.create(
    model='deepseek-reasoner',
    messages=[
      {"role": "system", "content": "你是一位大模型安全领域的分析专家，擅长分析攻击执行后的指标数据并给出一份理解深刻的分析报告。"},
      {"role": "user", "content": str}
    ],
    stream=False
  )
  return response.choices[0].message.content

def chatForReport(result):
  key = 'sk-919b0b8f52c24d43808aabadf2d30658'
  client = OpenAI(api_key=key, base_url='https://api.deepseek.com/v1')
  response = client.chat.completions.create(
    model='deepseek-reasoner',
    messages=[
      {"role": "system", "content": "你是一位大模型安全领域的分析专家，擅长分析攻击执行后的指标数据并给出一份理解深刻的分析报告。"},
      {"role": "user", 
      "content": '''我们通过攻击执行平台执行了攻击实验，首先让我来为你解释我们的攻击结果指标。
      首先这是一个字典形式的数据，最外层字典格式为：{"result": result, "globalConfig": globalConfig}。
      其中result中存储每一个攻击配置的攻击参数和结果，比如执行投毒攻击时的投毒率等参数、准确率（accuracy）等结果，
      globalConfig中存储了全局配置参数，如使用的模型、数据集等信息。
      result同样为一个字典，分别对应正常训练以及六种攻击类型。
      对于正常训练，键名为"normalTrain"，其余六种攻击类型分别为"AdvAttack"（对抗样本攻击）、"BackdoorAttack"（后门攻击）、"PoisoningAttack"（数据投毒攻击）、"RLMI"（模型反演攻击）、"FET"（梯度反演攻击）和"ModelStealingAttack"（模型窃取攻击）。
      可以注意到在六种攻击中前三种为安全攻击，后三种为隐私攻击。
      对于normalTrain键值对中的值，通常是[[a,b]]的格式，内部列表通常只有一个（这是由于常规训练只会执行一次），其中a对应任务准确率accuracy，b对应任务f1分数。
     对于AdvAttack键值对中的值，通常是[{'info':info, 'resultData': [a,b,c,d,e,f]},...]的格式，内部字典可能不只一个（这是由于对抗攻击可能执行多次），其中a对应攻击成功次数，b对应失败攻击次数，c对应被跳过的攻击次数，d对应攻击前任务准确率，e对应攻击后任务准确率，f对应攻击成功率。
     对于BackdoorAttack键值对中的值，通常是[{'info':info, 'resultData': [a,b,c,d,e,f,g]},...]的格式，内部字典可能不只一个（这是由于后门攻击可能执行多次），其中a对应投毒方数据集名称，b对应原始数据集准确率，c对应毒化数据集准确率，d对应PPL（后门攻击的困惑度），e对应USE（后门攻击的语义相似性），f对应后门攻击的语法正确性得分。
     对于PoisoningAttack键值对中的值，通常是[{'info':info, 'resultData': [a,b]},...]的格式，内部字典可能不只一个（这是由于投毒攻击可能执行多次），其中a对应任务准确率accuracy，b对应任务f1分数。
     对于RLMI键值对中的值，通常是[{'info':info, 'resultData': [a,b,c,d]},...]的格式，内部字典可能不只一个（这是由于模型反演攻击可能执行多次），其中a对应攻击阶段成功率，b对应攻击阶段词错误率，c对应推理阶段成功率，d对应推理阶段词错误率。
     对于FET键值对中的值，通常是[{'info':info, 'resultData': [[a,b,c,d,e,f,g,h], ..., [A,B,C,D,E,F]]},...]的格式，每一个内部字典表示一次完整FET执行（可能有多个FET执行，因此后续紧跟省略号），
     [a,b,c,d,e,f,g,h]对应FET攻击中每个Epoch，其中a对应原始序列，b对应反演攻击推理出的序列，c对应ROUGE-1得分，d对应ROUGE-2得分，e对应ROUGE-L得分，f对应词汇恢复率， g对应编辑距离（Edit Distance），h对应是否完全恢复的bool字段。“[a,b,c,d,e,f,g,h], ...”表示会有多个epoch。
     省略号后的[A,B,C,D,E,F]固定在最后一个，表示平均数据，即此次攻击中平均的ROUGE-1得分、ROUGE-2得分、ROUGE-L得分、词汇恢复率、编辑距离以及完全恢复率。
     对于ModelStealingAttack键值对中的值，通常是[{'info':info, 'resultData': {"iteration": iteration, "victim_acc": victim_acc, "steal_acc": steal_acc, "agreement": agreement}}, ...]的格式，攻击同样可能执行多次，每次生成一个字典。
     iteration对应迭代过程，为三元组（三元组的实际结构也是列表）的列表，每个三元组对应该次迭代后的训练损失、训练集的准确率、验证集准确率。最后的victim_acc对应受害者模型的准确率，steal_acc对应窃取者模型的准确率，agreement对应两者之间的相似度。
     可以注意到在前四种攻击中，没有包含迭代数据，后两种攻击的结果会更详尽一些，包含迭代过程的数据。每次攻击的字典的info表示了此次攻击的具体参数，如defenderEnabled表示是否启用防御（这里我们没有说明使用的防御策略，请你不要猜测，而是仅从防御和攻击的粒度来考虑，而不是考虑不同防御策略的影响）。
     info中的name仅仅是用户定义的名字，type表示攻击类型。另外，某些攻击的内容可能为空，这意味着没有测试该攻击，此时不再考虑对该攻击的分析。
     请严格按照以下结构生成Markdown格式的中文编码报告，回答中除了报告正文，请不要包含其他内容：

#### 1. 实验概览
- 从globalConfig提取关键配置：模型架构、数据集、训练参数等
- 列出所有执行的攻击类型及其分类（安全/隐私攻击）

#### 2. 基准性能分析
- 分析"normalTrain"结果：准确率和F1分数
- 与文献中的基准模型比较（如适用）

#### 3. 安全攻击分析（对抗样本、后门、投毒）
- **对抗样本攻击(AdvAttack)**:
	- 稍微解释该攻击
  - 计算平均攻击成功率
  - 分析攻击前后准确率变化
  - 统计攻击尝试分布（成功/失败/跳过）
  
- **后门攻击(BackdoorAttack)**:
	- 稍微解释该攻击
  - 对比毒化前后准确率
  - 评估攻击隐蔽性：困惑度、语义相似性、语法正确性
  - 生成毒化效果雷达图（概念描述）
  
- **数据投毒攻击(PoisoningAttack)**:
	- 稍微解释该攻击
  - 分析多次攻击的准确率和F1变化趋势
  - 计算攻击造成的平均性能下降

#### 4. 隐私攻击分析（模型反演、梯度反演、模型窃取）
- **模型反演攻击(RLMI)**:
	- 稍微解释该攻击
  - 对比攻击阶段和推理阶段指标
  - 成功率与词错误率的平衡分析
  
- **梯度反演攻击(FET)**:
	- 稍微解释该攻击
  - 提取最终综合指标：ROUGE-1、ROUGE-2、ROUGE-L、词汇恢复率、编辑距离、完全恢复率
  - 描述训练动态：ROUGE分数随epoch的变化趋势
  - 识别关键转折点（如分数突增的epoch）
  
- **模型窃取攻击(ModelStealingAttack)**:
	- 稍微解释该攻击
  - 对比受害者模型(victim_acc)和窃取模型(steal_acc)性能
  - 分析模型相似度(agreement)
  - 描述训练过程：损失/准确率随迭代的变化

## 5. 横向对比分析
### 安全攻击特性对比
| 评估维度         | {{#if AdvAttack}}对抗样本{{/if}} | {{#if BackdoorAttack}}后门{{/if}} | {{#if PoisoningAttack}}投毒{{/if}} |
|------------------|------------------|------------------|------------------|
| 性能影响         | {{#if AdvAttack}}{{impact_adv}}{{/if}} | {{#if BackdoorAttack}}{{impact_backdoor}}{{/if}} | {{#if PoisoningAttack}}{{impact_poison}}{{/if}} |
| 检测难度         | {{#if AdvAttack}}{{detect_adv}}{{/if}} | {{#if BackdoorAttack}}{{detect_backdoor}}{{/if}} | {{#if PoisoningAttack}}{{detect_poison}}{{/if}} |
| 缓解成本         | {{#if AdvAttack}}{{cost_adv}}{{/if}} | {{#if BackdoorAttack}}{{cost_backdoor}}{{/if}} | {{#if PoisoningAttack}}{{cost_poison}}{{/if}} |

**对比分析**：  
{{#if BackdoorAttack}}
- 后门攻击隐蔽性指数为{{stealth_index}}，超过行业平均{{industry_avg}}%  
{{/if}}
{{#if AdvAttack}}
- 对抗样本攻击成功率{{success_rate}}%，破坏强度超其他攻击{{comparative_impact}}%  
{{/if}}

### 隐私攻击特性对比
| 评估维度         | {{#if RLMI}}模型反演{{/if}} | {{#if FET}}梯度反演{{/if}} | {{#if ModelStealingAttack}}模型窃取{{/if}} |
|------------------|------------------|------------------|------------------|
| 信息质量         | {{#if RLMI}}{{info_quality_rlmi}}{{/if}} | {{#if FET}}{{info_quality_fet}}{{/if}} | {{#if ModelStealingAttack}}{{info_quality_steal}}{{/if}} |
| 实施复杂度       | {{#if RLMI}}{{complexity_rlmi}}{{/if}} | {{#if FET}}{{complexity_fet}}{{/if}} | {{#if ModelStealingAttack}}{{complexity_steal}}{{/if}} |
| 防御可行性       | {{#if RLMI}}{{defense_feasibility_rlmi}}{{/if}} | {{#if FET}}{{defense_feasibility_fet}}{{/if}} | {{#if ModelStealingAttack}}{{defense_feasibility_steal}}{{/if}} |

**对比分析**：  
{{#if ModelStealingAttack}}
- 模型窃取导致知识产权风险值达{{ip_risk}}，构成最高合规风险  
{{/if}}
{{#if FET}}
- 梯度反演完全恢复率{{full_recovery}}%，存在关键数据泄露隐患  
{{/if}}

### 关键风险评估
1. **最大业务威胁**：{{top_business_threat}}  
2. **最高合规风险**：{{top_compliance_risk}}  
3. **最紧急漏洞**：{{critical_vulnerability}}  

#### 6. 防御建议
- 针对每种攻击分析防御效果
- 推荐监控指标（实时检测攻击）
- 架构改进建议（增强鲁棒性）

> 附录：指标解释
...

### 输出要求
1. 使用专业术语但保持可读性
2. 关键发现用**加粗**强调
3. 对异常结果进行特别标注
4. 包含假设和推论（当数据不完整时）
5. 在结论部分总结最重要的发现
6. 在结尾用注释或其他格式给出相关指标的解释，只解释用到的指标，没用的不解释。
7. 使用utf-8编码
8. 分析的表格尽可能格式良好，比如用 | 符号分隔。 全文格式也请保持正确。
      
最后，我给出具体数据。我们的攻击执行结果指标数据是这样的（json格式）：''' + json.dumps(result) + '''\n请你按照以上要求生成一份报告。'''},
  ],
  stream=False
    )
  return response.choices[0].message.content



if __name__ == "__main__":
  # Example usage
  # info, result = get_attack_record(username='admin', timestamp=1737727113)
  # result = json.loads(result)
  # attackInfo = json.loads(info)

  # attackType = ['AdvAttack', 'BackdoorAttack', 'PoisoningAttack', 'RLMI', 'FET', 'ModelStealingAttack']
  # counters = [0, 0, 0, 0, 0, 0]
  # globalConfig = attackInfo[1]
  # attackInfo = attackInfo[0]
  # for info in attackInfo:
  #     tp = info['type']
  #     index = attackType.index(tp)
  #     result[tp][counters[index]] = {'info': info, 'resultData': result[tp][counters[index]]}
  #     counters[index] += 1

  # resultInfo = {"result": result, "globalConfig": globalConfig}
  # rep = chatForReport(resultInfo)
  # rep = chatForTest("你好，这是测试内容。请随意输出一段utf-8编码的中文内容。")
  

  f = open(os.path.join(lmsDir, "utils", 'testDeepseek.md'), "r", encoding='utf-8')
  rep = f.read()
  f.close()
  
  pdf = MarkdownPdf(toc_level=2, optimize=True)
  pdf.add_section(Section(rep))
  pdf.save(os.path.join(lmsDir, "logs",'reports', 'test.pdf'))
  # pdf.save("test.pdf")
  
	
