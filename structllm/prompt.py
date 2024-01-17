import structllm as sllm
import re
import json

class query_prompt():
    def __init__(self, question, model, add_retrieve, table_data, prompt_path):
        self.question = question
        self.model = model
        if self.model=="text-davinci-003":
            if add_retrieve:
                Top10nodes = sllm.align.Top10nodes(question, table_data)
                self.naive_prompt = f"Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question(you can use {Top10nodes} to generate parameters in query):\nQuestion: "+ self.question
            else:
                self.naive_prompt = "Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question:\nQuestion: "+ self.question
        else:
            if bool(re.compile(r'[\u4e00-\u9fa5]').search(question)): #换中文prompt
                with open(prompt_path, 'r') as json_file:
                    self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                    {
                      "role": "user",
                      "content": self.schema_Prompt(table_data, question)
                    }
                )
            else:
                with open(prompt_path, 'r') as json_file:
                    self.naive_prompt = json.load(json_file)
                self.naive_prompt.append(
                  {
                      "role": "user",
                      "content": self.schema_Prompt(table_data, question)
                  }
                )
                

    def schema_Prompt(self, table_data, question):
        relations, values = sllm.align.get_schema(table_data)
        
        tmp_prompt = str()
        for idx,item in enumerate(relations):
            if idx == 0:
                tmp_prompt += f"{item}:{values[idx]}"
            else:
                tmp_prompt += f"|{item}:{values[idx]}"

        prompt = f"Schema: {tmp_prompt}.\nQuestion: {question}"
        # import pdb;pdb.set_trace()
        return prompt


class kgqa_query_prompt():
    def __init__(self, question, model, add_retrieve, table_data, prompt_path):
        self.question = question
        self.model = model
        if self.model=="text-davinci-003":
            if add_retrieve:
                Top10nodes = sllm.align.Top10nodes(question, table_data)
                self.naive_prompt = f"Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question(you can use {Top10nodes} to generate parameters in query):\nQuestion: "+ self.question
            else:
                self.naive_prompt = "Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question:\nQuestion: "+ self.question
        else:
            with open(prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)
            self.naive_prompt.append(
                {
                    "role": "user",
                    "content": self.kgqa_schema_Prompt()
                }
            )
    
    def kgqa_schema_Prompt(self):
        
        prompt = f"Question: {self.question}"
        return prompt
    
class wqsp_query_prompt():
    def __init__(self, question, model, add_retrieve, table_data, prompt_path, First_step = None, Second_step = None):
        self.question = question
        self.model = model
        self.First_step = First_step
        self.Second_step = Second_step
        if self.model=="text-davinci-003":
            if add_retrieve:
                Top10nodes = sllm.align.Top10nodes(question, table_data)
                self.naive_prompt = f"Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question(you can use {Top10nodes} to generate parameters in query):\nQuestion: "+ self.question
            else:
                self.naive_prompt = "Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question:\nQuestion: "+ self.question
        else:
            with open(prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)

            prompt, TopicEntityID = self.wqsp_schema_Prompt()
            self.naive_prompt.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
            self.TopicEntityID = TopicEntityID
            # self.TopicEntityID = None
    
    def wqsp_schema_Prompt(self):
        # 第一个\n之前的为question 之后的为TopicEntityName
        question = self.question.split('\n')[0]
        TopicEntityName = self.question.split('\n')[1]
        TopicEntityID = self.question.split('\n')[2]
        prompt = f"Question: {question}\nTopicEntityName: {TopicEntityName}"
        if self.First_step:
            if type(self.First_step) == dict:
                relation_string = str()
                for key in self.First_step.keys():
                    item = self.First_step[key]
                    relation_string += f"{item}|"
                relation_string = relation_string[:-1]
                
                prompt += f"\nFirst_step: {relation_string}"
            else:
                prompt += f"\nFirst_step: {self.First_step}"
        

        if self.Second_step:
            if type(self.Second_step) == dict:
                relation_string = str()
                for key in self.Second_step.keys():
                    item = self.Second_step[key]
                    relation_string += f"{item}|"
                relation_string = relation_string[:-1]
                prompt += f"\nSecond_step: {relation_string}"
            else:
                prompt += f"\nSecond_step: {self.Second_step}"

        return prompt, TopicEntityID


class temp_query_prompt():
    def __init__(self, question, model, add_retrieve, table_data, prompt_path, relation_list, annotation):
        self.question = question
        self.model = model
        self.relation_list = relation_list
        self.annotation = annotation
        if self.model=="text-davinci-003":
            if add_retrieve:
                Top10nodes = sllm.align.Top10nodes(question, table_data)
                self.naive_prompt = f"Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question(you can use {Top10nodes} to generate parameters in query):\nQuestion: "+ self.question
            else:
                self.naive_prompt = "Now you are learning to write conditional graph query composed of functions for questions in natural language.  The format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\", and you can use the Set function and Calculator function in combination.\n\nSet function:\n - set_union(set1, set2)\n - set_intersection(set1, set2)\n - set_difference(set1, set2)\n - set_negation(set)\n\nCalculator function:\n - sum(set1)\n - mean(set1)\n - max(set1)\n - min(set1)\n - count(set1)\n\nYour task is to break down the original natural language problem step by step, the format of the query is \"\"\"get_infomation(relation, head_entity, tail_entity, key, value, condition)\"\"\" mentioned above. Following are some examples.\n\nQuestion: What is the capital of China?\nStep1: Find the capital of China\nQuery1: \"get_information(relation='capital', head_entity='China')\"\n\nQuestion: Which country's capital is Beijing?\nStep1: Find the country whose capital is Beijing\nQuery1: \"get_information(relation='capital', tail_entity='Beijing')\"\n\nQuestion: Who is born in 2020?\nStep1: Find people who is born in 2020\nQuery1: \"get_information(key='born in', value='2020')\"\n\nQuestion: What was the sum of the market value of the company of the most valuable company in China and the most valuable in America in 2020?\nStep1: Find companies that are in China and in 2020\nQuery1: \"get_information(relation='country', tail_entity='China', key='year', value='2020')\"\nStep2: Find companies that are in America and in 2020\nQuery2: \"get_information(relation='country', tail_entity='America', key='year', value='2020')\"\nStep3: Find market value of output_of_query1 in 2020\nQuery3: \"get_information(relation='market value', head_entity='output_of_query1', key='year', value='2020')\"\nStep4: Calculate max of output_of_query3\nQuery4: \"max(set1='output_of_query3')\"\nStep5: Find market value of output_of_query2 in 2020\nQuery5: \"get_information(relation='market value', head_entity='output_of_query2', key='year', value='2020'))\"\nStep6: Calculate max of output_of_query5\nQuery6: \"max(set1='output_of_query5')\"\nStep7: Calculate sum of output_of_query4 and output_of_query6\nQuery7:  \"sum(set1='output_of_query4', set2='output_of_query6')\"\n\nQuestion: Which manufacturers can produce power output greater than 100bhp, or a maximum speed greater than 200?\nStep1: Find power output greater than 100bhp\nQuery1: \"get_information(relation='power output')>100\"\nStep2:  Find maximum speed greater than 200\nQuery2:  \"get_information(relation='max speed')>200\"\nStep3:  Get union of output_of_query1 and output_of_query2\nQuery3:  \"set_union(set1='output_of_query1', set1='output_of_query2')\"\n\nQuestion: What is the most expensive item in the Electronics category?\nStep1: Find all items under the electronic category\nQuery1: \"get_information(relation='category', tail_entity='Electronics')\" \nStep2: Find the price of output_of_query1\nQuery2: \"get_information(relation='price', head_entity='output_of_query1')\"\nStep3: Calculate the max of output_of_query2\nQuery3: \"max(set1='output_of_query2')\"\n\nWhat should be the query for the following question:\nQuestion: "+ self.question
        else:
            with open(prompt_path, 'r') as json_file:
                self.naive_prompt = json.load(json_file)

            prompt = self.temp_schema_Prompt()
            self.naive_prompt.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
    
    def temp_schema_Prompt(self):
        # 第一个\n之前的为question 之后的为TopicEntityName
        question = self.question
        relation_list = self.relation_list
        annotation = self.annotation
        prompt = f"Question: {question}\nRelations: {relation_list}\nannotation: {annotation}"
        return prompt

class retrieve_prompt():
    def __init__(self, head_entity, retrieve, CG_relations):
        self.head_entity = head_entity
        self.retrieve = retrieve
        self.CG_relations = CG_relations
        if bool(re.compile(r'[\u4e00-\u9fa5]').search(head_entity)): #换中文prompt
            self.naive_prompt = f"给定一个实体: “{head_entity}”，以及可能的实体类别: {CG_relations}. 请选择“{head_entity}”最可能属于的类别名称："
        else:
            self.naive_prompt = [
              {
                "role": "user",
                # "content": f"Given an:\"{head_entity}\", and the result of this entity retrieve: {retrieve}, and all candidate: {CG_relations}. Please select {head_entity} is most likely to belong to the category of the name:"
                # "content": f"Given an:\"{head_entity}\" and all candidate: {CG_relations}. Please select {head_entity} is most likely to belong to the category of the name:"
                "content": f"Please provide the correct relation type for \"{head_entity}\" and the options are: {CG_relations}. The relation type:"
              }
            ]
            # print(self.naive_prompt[0]["content"])
            
            
            
            

class store_prompt():
    def __init__(self,information) -> None:
        self.information = information
        self.naive_prompt = [
            {
                'role':'system',
                'content':'You are specialist in knowledge graph construction. Given a statement, your task is construct a binary relational fact or hyper-relational fact based on the statement. Here are some demonstrations.'
            },
            {
                'role':'user',
                'content':'Statement: John Biden is the President of the United States.'
            },
            {
                'role':'assistant',
                'content':'Binary relational fact(John Biden, president of, the United States)'
            },
            {
                'role':'user',
                'content':"Statement: Mike's telephone number is 123456789."
            },
            {
                'role':'assistant',
                'content':'Binary relational fact(Mike, telephone number, 123456789)'
            },
            {
                'role':'user',
                'content':'Statement: Marie Curie was awarded the Nobel Prize in Physics in 1903, together with Pierre Curie.'
            },
            {
                'role':'assistant',
                'content':'Hyper-relational fact(Marie Curie, award-received, Nobel Prize in Physics, point-in-time:1903, together-with:Pierre Curie)'
            },
            {
                'role':'user',
                'content':'Statement: Barack Obama replaced George W. Bush as the 44th US president on 20 January 2009, and was replaced by Donald Trump on 20 January 2017.'
            },
            {
                'role':'assistant',
                'content':'Hyper-relational fact(Barack Obama, position held, President of United States, start time:20 January 2009, replaces:George W. Bush, series ordinal:44, end time:20 January 2017, replaced by:Donald Trump)'
            },
            {
                'role':'user',
                'content':'Statement: ' + information
            }
        ]
        

class relation_prompt():
    def __init__(self, extracted_relation, question, CG_relations):
        self.extracted_relation = extracted_relation
        self.question = question
        self.CG_relations = CG_relations
        if bool(re.compile(r'[\u4e00-\u9fa5]').search(extracted_relation)): #换中文prompt
            self.naive_prompt = f"给定全部关系类型:\"{CG_relations}\", 请将问题:\"{question}\"中提取出的关系\"{extracted_relation}\"对应到最可能的关系类型："
        else:
            self.naive_prompt=[
              {
                "role": "user",
                "content": f"Given relation types: {CG_relations}, please align the \"{extracted_relation}\" mentioned in  \"{question}\" to the possible relation type:"
              },
            ]
            # self.naive_prompt = f"Given relation types: {CG_relations},  please align the \"{extracted_relation}\" mentioned in the question \"{question}\" to the most possible relation type: "
 
class result_prompt():
    
    def get_query_and_mid_output(self, list_query, mid_output, cgdata):
        result = str()
        for idx,query in enumerate(list_query):
            result += query+','
            key = f'output_of_query{idx+1}'
            result += key+": "
            try:
                tmp_result = mid_output[key]
                dic = set(cgdata.id2node[idx] for idx in tmp_result)
                result += str(dic)
            except:
                result += "None"
        return result

    def __init__(self, question, list_query, mid_output, cgdata, res=None):
        self.question = question
        tmp_reason = self.get_query_and_mid_output(list_query, mid_output, cgdata)
        if mid_output==set() or mid_output==None or mid_output=='None':
            self.naive_prompt = self.question
        else: 
            self.naive_prompt = f"Here is a question: \"\"\"{self.question}\"\"\". There is query to help answer this question that \"\"\"{tmp_reason}\"\"\". Result for the query is {res}. Thus the answer for \"\"\"{self.question}\"\"\" is "