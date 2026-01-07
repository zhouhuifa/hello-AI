from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

from util.my_llm import llm_gpt

# ICL In Context Learning

# 步骤一 提供示例
examples = [
    {
        "question":"默罕默德·阿里和艾伦·图灵谁活得更久？",
        "answer":"""是否需要后续问题：是。
        后续问题：默罕默德·阿里去世时多大？
        中间答案：默罕默德·阿里去世时74岁。
        后续问题：艾伦·图灵去世时多大？
        中间问题：艾伦·图灵去世时41岁。
        所以最终答案是：默罕默德·阿里""",
    },{
        "question":"乔治华盛顿的外祖父是谁？",
        "answer":"""是否需要后续问题：是。
        后续问题：乔治华盛顿的母亲是谁？
        中间答案：乔治华盛顿的母亲是玛丽鲍尔华盛顿
        后续问题：玛丽鲍尔华盛顿的父亲是谁？
        中间问题：玛丽鲍尔华盛顿的父亲是约瑟夫鲍尔
        所以最终答案是：约瑟夫鲍尔""",
    }
]

base_template = PromptTemplate.from_template("问题：{question}\n{answer}")

# 步骤二 创建FewShotPromptTemplate实例
final_template = FewShotPromptTemplate(
    examples=examples, # 传入示例列表
    example_prompt=base_template, # 指定单个示例的提示模版
    suffix="问题：{input}",   # 最后追加的问题模版
    input_variables=["input"]   # 指定输入变量
)

# 问题模版和模型组成链
chain = final_template | llm_gpt
# res = chain.invoke({"input":"巴伦特朗普的父亲是谁？"})
res = chain.invoke({"input":"中国古代历史上，唐朝和宋朝那个延续的时间最长？"})
print(res)

