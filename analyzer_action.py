import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

import json
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

PROMPT_TEMPLATE = """
你是一位数据分析助手，你的回应内容取决于用户的请求内容。

1. 对于文字回答的问题，按照这样的格式回答：
   {"answer": "<你的答案写在这里>"}
例如：
   {"answer": "订单量最高的产品ID是'MNWC3-067'"}

2. 如果用户需要一个表格，按照这样的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用户的请求适合返回条形图，按照这样的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用户的请求适合返回折线图，按照这样的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用户的请求适合返回散点图，按照这样的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我们只支持三种类型的图表："bar", "line" 和 "scatter"。

请将所有输出作为JSON字符串返回。请注意要将"columns"列表和数据列表中的所有字符串都用双引号包围。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

你要处理的用户请求如下： 
"""


def dataframe_agent(openai_api_key, df, query):
    # 检查 API 密钥
    if not openai_api_key:
        raise ValueError("OpenAI API key is not set in environment variables")

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key,
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )

    # 创建 agent，添加必要的安全参数
    agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,  # 明确允许代码执行
        max_execution_time=30,  # 设置最大执行时间为30秒
        max_iterations=100,  # 设置最大迭代次数
    )

    prompt = PROMPT_TEMPLATE + query

    try:
        response = agent.invoke({"input": prompt})
        output = response.get("output", response)

        # 处理输出
        try:
            if isinstance(output, str):
                return json.loads(output)
            elif isinstance(output, dict):
                return output
            else:
                return {"answer": str(output)}
        except json.JSONDecodeError:
            return {"answer": str(output)}

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# def main():
#     try:
#         # 获取 API 密钥
#         openai_api_key = os.getenv("OPENAI_API_KEY")
#         if not openai_api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment variables")
#
#         # 读取 CSV 文件
#         df = pd.read_csv("personal_data.csv")
#
#         # 执行分析
#         result = dataframe_agent(openai_api_key, df, "数据里出现最多的职业是什么?")
#
#         # 打印结果
#         print(json.dumps(result, ensure_ascii=False, indent=2))
#
#     except FileNotFoundError:
#         print("Error: personal_data.csv file not found")
#     except Exception as e:
#         print(f"Error: {str(e)}")
#
#
# if __name__ == "__main__":
#     main()