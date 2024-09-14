import os
import re
import pandas as pd
from config.common_config import *
import time
from lmdeploy.serve.openai.api_client import APIClient
import concurrent.futures

def process_single_choice(value, value_domain):
    return value if value in value_domain else "未提及"

def process_multi_choice(values, value_domain):
    # Split the comma-separated string into a list, process it, and then join back into a string
    values_list = values.split(',')
    processed_values = [value for value in values_list if value in value_domain]
    return ','.join(processed_values)

class LLMPredict(object):
    """自动保存最新模型
    """
    def __init__(self):
        self.api_client = APIClient(llmdeploy_url)
        self.model_name = self.api_client.available_models[0]

    def pred_res(self,Instruction,Input):
        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{Instruction}\n\n{Input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        completion = list(self.api_client.completions_v1(model=self.model_name, prompt=prompt, temperature=temperature))[0]
        answer = completion['choices'][0]['text']
        token_num = completion['usage']['completion_tokens']
        return answer,token_num
    
    # 硬控输出结果
    def get_result(self,project_name,field_en,field,context):
        prompt = ''
        
        medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=project_name)
        row = medical_logic.loc[medical_logic['字段英文名称'] == field_en].to_dict('records')[0]
        # field = row['字段名'] if '字段名' in row and pd.notnull(row['字段名']) else field
        special_requirement = row['特殊要求'] if '特殊要求' in row and pd.notnull(row['特殊要求']) else ''
        if row['值域类型'] == '多选':
            prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择提到的所有内容。{special_requirement}"""
        elif row['值域类型'] == '单选':
            prompt = f"##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择1个。{special_requirement}"
        elif row['值域类型'] == '提取':
            row['字段名'] = row['字段名'].replace('大小1','大小')
            prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请提取文中对应的值。{special_requirement}"""
        else:
            return ''
        
        print('prompt',prompt)
        start_time = time.time()
        res, token_num = self.pred_res(prompt,context) #if self.model != 'no enough GPU' else 'no enough GPU',0
        end_time = time.time()
        print('token_num',token_num)
        infer_time = round(end_time-start_time,6)
        per_token = infer_time/token_num
        print('model infer time~~',infer_time)
        print('per_token',per_token)
        print('res',res)
        if row['值域类型']== '单选':
            processed_result = process_single_choice(res, row['值域'])
        elif row['值域类型'] == '多选':
            processed_result = process_multi_choice(res, row['值域'])
        else:
            processed_result = res
        return res, token_num


    # 硬控输出结果
    def get_batch_result(self, data_list):
        start_time = time.time()
        project_name = data_list[0]["project"]
        medical_logic = pd.read_excel('./config/medical_logic.xlsx', sheet_name=project_name)
        prompts, contexts = [], []
        for data_dict in data_list:
            project_name,field_en,field,context = data_dict["project"],data_dict["field_en"],data_dict["field"],data_dict["raw_text"]
            row = medical_logic.loc[medical_logic['字段英文名称'] == field_en].to_dict('records')[0]
            # field = row['字段名'] if '字段名' in row and pd.notnull(row['字段名']) else field
            special_requirement = row['特殊要求'] if '特殊要求' in row and pd.notnull(row['特殊要求']) else ''
            if row['值域类型'] == '多选':
                prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择提到的所有内容。{special_requirement}"""
            elif row['值域类型'] == '单选':
                prompt = f"##结构化任务##根据下文中信息，判断{field}是什么？请在值域【{row['值域']}】中选择1个。{special_requirement}"
            elif row['值域类型'] == '提取':
                row['字段名'] = row['字段名'].replace('大小1','大小')
                prompt = f"""##结构化任务##根据下文中信息，判断{field}是什么？请提取文中对应的值。{special_requirement}"""
            else:
                return ''
            print(row['值域类型'] ,'prompt',prompt)
            prompts.append(prompt)
            contexts.append(context)
        res, token_num = self.pred_batch_res(prompts,contexts) # if self.model != 'no enough GPU' else 'no enough GPU',0
        end_time = time.time()
        infer_time = round(end_time - start_time, 6)
        per_token = infer_time / token_num
        print('model infer time~~', infer_time)
        print('per_token', per_token)
        print('res',res)

        # Process the results based on value domain type
        processed_results = []
        for i, row in enumerate(data_list):
            value_domain_type = medical_logic.loc[medical_logic['字段英文名称'] == row["field_en"], "值域类型"].values[0]
            print('value_domain_type',value_domain_type)
            value_range =  medical_logic.loc[medical_logic['字段英文名称'] == row["field_en"], "值域"].values[0]
            if value_domain_type == '单选':
                processed_result = process_single_choice(res[i], value_range)
            elif value_domain_type == '多选':
                processed_result = process_multi_choice(res[i], value_range)
            else:
                processed_result = res[i]

            processed_results.append(processed_result)

        return processed_results, token_num


    def pred_batch_res(self, prompts, contexts):
        results = [None] * len(prompts)  # 初始化一个与prompts大小一致的结果列表
        total_token_num = 0
        
        # 定义一个内部函数用于并发调用 pred_res
        def call_pred_res(index, prompt, context):
            return index, self.pred_res(prompt, context)
        
        # 使用 ThreadPoolExecutor 进行并发处理
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交所有的任务给线程池，并将索引一起传递
            future_to_res = {executor.submit(call_pred_res, i, prompts[i], contexts[i]): i for i in range(len(prompts))}
            
            # 收集所有的结果
            for future in concurrent.futures.as_completed(future_to_res):
                try:
                    index, (answer, token_num) = future.result()
                    results[index] = answer  # 将结果放在对应的索引位置
                    total_token_num += token_num
                except Exception as e:
                    # 处理调用中的异常情况
                    print(f"Error processing prompt {future_to_res[future]}: {e}")
        
        # 返回结果列表和总的 token 数量
        return results, total_token_num


# if __name__ == '__main__':
#     fd_pred = LLMPredict()
#     for i in range(5):
#         start_time = time.time()
#         data_list = [{
#                 "raw_text": """患者父親有前列腺癌史""",
#                 "field": "疾病名稱",
#                 "project": "澳门镜湖",
#                 "field_en": "FDNAM"
#             },
#             {
#                 "raw_text": "右側輸尿管結石術後反复血尿1月",
#                 "field": "症狀名稱",
#                 "project": "澳门镜湖",
#                 "field_en": "CCPSNAM"
#             },
#             {
#                 "raw_text": "月經婚育史(年齡&lt;=55y): LMP:2023/3/14,已婚未育.",
#                 "field": "末次月經日期",
#                 "project": "澳门镜湖",
#                 "field_en": "MOLMPDTC"
#             }]
#         res = fd_pred.get_batch_result(data_list)
#         print('final res',res)
#         end_time = time.time()
#         print('cost time:', round(end_time-start_time, 2))
