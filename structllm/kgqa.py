import re
import structllm as sllm
import openai

def kgqa(args, question, table_data, relations): 

    llm = sllm.llm.gpt(model = args.model, key = args.key)
    result_list = []

    max_retries = 3         # 最大重试次数
    retry_count = 0         # 当前重试次数
    SC_Num = args.SC_Num    # self consistency
    result_list = []    
    total_num = 0           # 防止卡死
        
    while retry_count < max_retries and SC_Num >0 and total_num < max_retries*SC_Num:
        retry_count += 1

        print(f"Retry_count:{retry_count}, Num of SC:{SC_Num}, Total_numL:{total_num}")

        '''question -> query'''
        query_prompt = sllm.prompt.kgqa_query_prompt(question, args.model, args.add_retrieve, table_data, args.prompt_path) # 得到query的prompt
        print(query_prompt.naive_prompt[-1])

        responses = llm.get_response(query_prompt.naive_prompt, flag = 1, num = SC_Num)        
            
        for response in responses:
            try:
                response = response.message["content"]
                print("      ###1.generated query_text:", response) # 查看query
                # Step2: Get target_type from response
                type2id, target_type = sllm.align.get_target_type(response, table_data) # 得到query的类型
                print("      ###2.generated target_type:", target_type) # 查看query

                # Step3: parameter retrieval and replacement
                text_query, id_query, step_query = sllm.align.MetaQA_text2query(args, response, question, table_data, relations) # text_query:进行node2id前的query，id_query:进行node2id后的query
                print("      ###3.retrieved parameters(node):", text_query)
                print("      ###3.retrieved parameters(id):", id_query)

                '''query -> result'''
                # 执行query
                print("      ###4.excute process:")
                if target_type == None: res, mid_output = table_data.excute_query(args, id_query, target_type=None, node_query=text_query, task=step_query, question=question)
                else: res, mid_output = table_data.excute_query(args, id_query, target_type=target_type[0], node_query=text_query, task=step_query, question=question)
                
            except openai.error.InvalidRequestError as e: # 非法输入 '$.input' is invalid. query返回结果为：请输入详细信息等 
                print(e)
                total_num += 1
                continue

            except IndexError as e: # 得不到正确格式的query: set1=(fastest car)
                print(e)
                total_num += 1 # 防止卡死
                continue

            except ValueError as e: # maximum context length
                print(e)
                continue
            
            total_num += 1 # 可得到结果

            if res == None or res == [] or res == set() or res == [set()] or res == dict() or res == [set([])] or res == ['None'] or res == ['none'] or (type(res)==str and "[line_" in res ):
                print("      ###5.excute result is None, retry...")
                if retry_count >= max_retries: result_list.append(res)
                continue
            else:
                print("      ###5.excute result:",res)
                SC_Num -= 1
                result_list.append(res)

    while len(result_list) < args.SC_Num: result_list.append("0")
    print("###6.final result", result_list)
    return result_list