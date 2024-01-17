import re
import structllm as sllm
import random
import ast
# from llm import Query

"""
following codes are from API_generate.ipynb
"""

from sentence_transformers import SentenceTransformer,util
import pandas as pd
import numpy as np
import torch
import re
import openai
# from DataProcess import APIData, APIDataReader

class M3ERetriever:
    def __init__(self,corpus) -> None:
        self.retrieve_model = SentenceTransformer('moka-ai/m3e-base')
        self.corpus = corpus
    def get_topk_candidates(self,topk,query):
        # query is a list

        # Corpus of documents and their embeddings
        corpus_embeddings = self.retrieve_model.encode(self.corpus)
        # Queries and their embeddings
        queries_embeddings = self.retrieve_model.encode(query)
        # Find the top-k corpus documents matching each query
        cos_scores = util.cos_sim(queries_embeddings, corpus_embeddings)
        # hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=topk)
        all_query_candidate_api_index = []
        for i in range(len(cos_scores)):
            hits = torch.argsort(cos_scores[i],descending=True)[:topk]
            all_query_candidate_api_index.append(hits.tolist())
        return all_query_candidate_api_index
    
    def count_accuracy(self,label,candidate):
        assert len(label)==len(candidate)

        topk_count = 0
        # hit = [0]*30
        # count = [0]*30
        for i in range(len(label)):
            # count[label[i]] += 1
            if label[i] in candidate[i]:
                topk_count += 1
                # hit[label[i]] += 1
        accuracy = topk_count/len(label)
        return accuracy

class OpenAIRetriever:
    def __init__(self,corpus) -> None:
        self.corpus = corpus
    def get_embedding(self, text, model="text-embedding-ada-002"):
        return openai.Embedding.create(input = text, model=model)['data']
    def get_topk_candidates(self,topk,query):
        # query is a list

        # Corpus of documents and their embeddings
        api_corpus_embeddings = self.get_embedding(self.corpus)
        api_corpus_embeddings_list = []
        for i in range(len(api_corpus_embeddings)):
            api_corpus_embeddings_list.append(api_corpus_embeddings[i]['embedding'])
        # Queries and their embeddings
        queries_embeddings = self.get_embedding(query)
        queries_embeddings_list = []
        for i in range(len(queries_embeddings)):
            queries_embeddings_list.append(queries_embeddings[i]['embedding'])
        # Find the top-k corpus documents matching each query
        cos_scores = util.cos_sim(queries_embeddings_list, api_corpus_embeddings_list)
        # hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=topk)
        all_query_candidate_api_index = []
        for i in range(len(cos_scores)):
            hits = torch.argsort(cos_scores[i],descending=True)[:topk]
            all_query_candidate_api_index.append(hits.tolist())
        return all_query_candidate_api_index
    def count_accuracy(self,label,candidate):
        assert len(label)==len(candidate)

        topk_count = 0
        # hit = [0]*30
        # count = [0]*30
        for i in range(len(label)):
            # count[label[i]] += 1
            if label[i] in candidate[i]:
                topk_count += 1
                # hit[label[i]] += 1
        accuracy = topk_count/len(label)

        return accuracy

def get_target_type(response,cgdata):
    pattern = r'\{(.*?)\}'  # 正则表达式模式，匹配大括号中的内容
    matches = re.findall(pattern, response)  # 查找所有匹配的内容
    if matches:
        result = matches[0]  # 获取第一个匹配的内容
        _, CG_relations = get_entitise_relations(cgdata)
        retriever = OpenAIRetriever(CG_relations)
        top1_api = retriever.get_topk_candidates(1,result)
        label_rel = [CG_relations[char[0]] for char in top1_api]
        # print(top1_api)
        # print(label_rel)
    else:
        top1_api = None
        label_rel = None
    return top1_api, label_rel

def update_head_entity(args, head_entity, CG_relations):
    retriever = OpenAIRetriever(CG_relations)
    top5_api = retriever.get_topk_candidates(5,head_entity)
    label_rel = [CG_relations[char] for char in top5_api[0]]
    retrieve_prompt = sllm.prompt.retrieve_prompt(head_entity, label_rel, CG_relations)
    llm = sllm.llm.gpt(model = args.model, key = args.key)
    response = llm.get_response(retrieve_prompt.naive_prompt)
    # cleaned_string = response.lstrip().replace("'", "") # 删除所有单引号'和前置空格
    # cleaned_string = response.strip().replace("'", "").replace('"', "")
    res = label_rel[0] # 默认为第一个
    for relation in CG_relations: # 寻找lable relation
        if response.find(relation) != -1:
            res = relation
            break
    return "key='"+res+"', value"

def get_relation_alignment(args, texts, question, CG_relations):
    result = []
    for text in texts:
        match = re.search(r"relation='(.*?)'", text)
        # 检查是否找到匹配项并提取内容
        if match:
            extracted_relation = match.group(1)
            # print(extracted_relation)
        else: 
            result.append(text)
            continue

        retrieve_prompt = sllm.prompt.relation_prompt(extracted_relation, question, CG_relations)
        llm = sllm.llm.gpt(model = args.model, key = args.key)
        response = llm.get_response(retrieve_prompt.naive_prompt)
        for relation in CG_relations:
            if response.find(relation) != -1:
                res = relation
                break
        try:
            result.append(text.replace("'"+extracted_relation+"'", "'"+res+"'"))
        except UnboundLocalError: # 没有找到匹配项不替换
            result.append(text)
    return result

# query = sllm.align.text2query(response, table_data["1-10015132-11"])
def text2query(args, text, question, cgdata):
    """translate text into formal queries
    output: query class
    """
    pattern_step = r"(Step.*)"
    step_text = re.findall(pattern_step, text) # 得到list形式的query
    print(f"step_text:{step_text}")

    pattern_query = r'Query\d+\s*:\s*"(.*?)"'
    text = re.findall(pattern_query, text) # 得到list形式的query
    print(f"text:{text}")

    CG_entities, CG_relations = get_entitise_relations(cgdata)

    # update head entity into key-value pair
    if text[0].find("head_entity") != -1:
        head_entity = re.findall(r'head_entity=\'([^"]*)\'', text[0])
        text[0] = text[0].replace("head_entity", update_head_entity(args, head_entity[0], CG_relations))
    
    # update relation with GPT
    # text = get_relation_alignment(args, text, question, CG_relations)
    # print(text)
    
    Re_entities, Re_relations = get_parameters(text) #参数提取
    print(f"Re_entities:{Re_entities}")
    print(f"Re_relations:{Re_relations}")
    # retrieve entities
    if Re_entities == []:
        label_ent = []
    else:
        retriever = OpenAIRetriever(CG_entities)
        top1_api = retriever.get_topk_candidates(1,Re_entities)
        label_ent = [CG_entities[char[0]] for char in top1_api]
    
    # retrieve relations
    if Re_relations == []:
        label_rel = []
    else:
        retriever = OpenAIRetriever(CG_relations)
        top1_api = retriever.get_topk_candidates(1,Re_relations)
        label_rel = [CG_relations[char[0]] for char in top1_api]

    text_query, id_query = replace_query(text, Re_entities, label_ent, Re_relations, label_rel, cgdata.node2id)
    # import pdb; pdb.set_trace();
    # text_query, id_query = replace_query(text, Re_entities, label_ent, cgdata.node2id)
    # ==> text_query: get_information(relation=Position, tail entity=Butler CC (KS))
    # ==> id_query:   get_information(relation=1, tail entity=2)
    return text_query, id_query, step_text

def replace_query(responses, Re_entities, label_ent, Re_relations, label_rel, node2id):
    # 创建映射字典
    Re_list = Re_entities + Re_relations
    label_list = label_ent + label_rel
    replace_dict = dict(zip(Re_list, label_list))

    # 使用正则表达式查找要替换的元素
    pattern = '|'.join(re.escape("'"+key+"'") for key in Re_list)

    def replace_match(match):
        matched_text = match.group()
        element = matched_text[1:-1]  # 去掉单引号
        replacement = replace_dict.get(element, element)
        return f"'{replacement}'"

    # text_query = [re.sub(pattern, lambda x: replace_dict[x.group()], response) for response in responses]
    text_query = [re.sub(pattern, replace_match, response) for response in responses]
    # text_query = re.sub(pattern, lambda x: replace_dict[x.group()], responses)
    
    # 创建映射字典
    Re_list = label_list
    # import pdb; pdb.set_trace();
    label_list = [ node2id[node] for node in Re_list]
    replace_dict = dict(zip(Re_list, label_list))

    # 使用正则表达式查找要替换的元素
    pattern = '|'.join(re.escape("'"+key+"'") for key in Re_list)

    
    # id_query = [re.sub(pattern, lambda x: replace_dict[x.group()], response) for response in text_query]
    id_query = [re.sub(pattern, replace_match, response) for response in text_query]

    return text_query, id_query

def extract_query(text):
    function = None 
    pass 

def getStr(x): return x if type(x)==str else str(x)

def get_parameters(responce_from_api):
    entities = []
    relations = []
    # pattern = r"'(.*?)'"
    rel_pattern = r"(?:relation|key)\s*=\s*'(.*?)'(?:,|\))"
    ent_pattern = r"(?:entity|value)\s*=\s*'(.*?)'(?:,|\))"

    for item in responce_from_api:
        matches_ent = re.findall(ent_pattern, item)
        matches_rel = re.findall(rel_pattern, item)
        for match in matches_ent:
            if match[:6] != 'output' and getStr(match) not in entities and getStr(match) != '':
                entities.append(getStr(match))
        for match in matches_rel:
            if match[:6] != 'output' and getStr(match) not in relations and getStr(match) != '':
                relations.append(getStr(match))

    return entities, relations

def get_entitise_relations(cgdata):
    entities = set()
    relations = set()
    for h,r,t in cgdata.triples:
        if h == 'row_number': continue

        if t == '[0]':
            relations.add(getStr(r))
            entities.add(getStr(h))
        else:
            entities.add(getStr(r))
            entities.add(getStr(t))
    
    CG_entities = [ item for item in list(entities) if item[0:6] != '[line_' and item!='']
    return CG_entities, list(relations)

def get_all_nodes(cgdata):
    nodes = set()
    for h,r,t in cgdata.triples:
        if t == '[0]':
            nodes.add(getStr(r))
            nodes.add(getStr(h))
        else:
            nodes.add(getStr(r))
            nodes.add(getStr(t))
    return list(nodes)

def Top10nodes(question, table_data):
    text = get_all_nodes(table_data)
    candidate = [question]
    # import pdb; pdb.set_trace();
    text = [node if type(node)==str else str(node) for node in text]
    # print(text)
    # print(candidate)
    retriever = OpenAIRetriever(text)
    
    top10_api = retriever.get_topk_candidates(10,candidate)
    label_ent = [text[idx] for idx in top10_api[0]]
    return label_ent

def get_schema(cgdata):
    schema_dict = dict()
    for h,r,t in cgdata.triples:
        if t == '[0]': continue # (line_0, relation,[0])
        # (relation, tail, head)
        if h in schema_dict:
            schema_dict[h].append(r)
        else:
            schema_dict[h] = [r]

    keys = []
    values = []
    for key, value_list in schema_dict.items():
        keys.append(key)
        values.append(random.choice(value_list))

    return keys, values


# KGQA MetaQA

def MetaQA_text2query(args, text, question, cgdata, CG_relations):
    """translate text into formal queries
    output: query class
    """
    pattern_step = r"(Step.*)"
    step_text = re.findall(pattern_step, text) # 得到list形式的query
    print(f"step_text:{step_text}")

    pattern_query = r'Query\d+\s*:\s*"(.*?)"'
    text = re.findall(pattern_query, text) # 得到list形式的query
    print(f"text:{text}")

    if CG_relations == None:
        _, CG_relations = get_entitise_relations(cgdata)
    else:
        CG_relations = list(CG_relations)
    # CG_entities, CG_relations = get_entitise_relations(cgdata)
        
    # print(CG_relations)
    # import pdb; pdb.set_trace();

    # # update head entity into key-value pair
    # if text[0].find("head_entity") != -1:
    #     head_entity = re.findall(r'head_entity=\'([^"]*)\'', text[0])
    #     text[0] = text[0].replace("head_entity", update_head_entity(args, head_entity[0], CG_relations))
    
    # update relation with GPT
    # text = get_relation_alignment(args, text, question, CG_relations)
    # print(text)
    
    Re_entities, Re_relations = get_parameters(text) #参数提取
    print(f"Re_entities:{Re_entities}")
    print(f"Re_relations:{Re_relations}")
    # # retrieve entities
    # if Re_entities == []:
    #     label_ent = []
    # else:
    #     retriever = OpenAIRetriever(CG_entities)
    #     top1_api = retriever.get_topk_candidates(1,Re_entities)
    #     label_ent = [CG_entities[char[0]] for char in top1_api]
    
    # retrieve relations
    if Re_relations == []:
        label_rel = []
    else:
        retriever = OpenAIRetriever(CG_relations)
        top1_api = retriever.get_topk_candidates(1,Re_relations)
        label_rel = [CG_relations[char[0]] for char in top1_api]

    text_query, id_query = replace_query(text, Re_entities, Re_entities, Re_relations, label_rel, cgdata.node2id)
    
    return text_query, id_query, step_text


# KGQA WQSP

def WQSP_text2query(args, text, question, cgdata, CG_relations=None, TopicEntityID=None):
    """translate text into formal queries
    output: query class
    """
    pattern_step = r"(Step.*)"
    step_text = re.findall(pattern_step, text) # 得到list形式的query
    print(f"step_text:{step_text}")

    pattern_query = r'Query\d+\s*:\s*"(.*?)"'
    text = re.findall(pattern_query, text) # 得到list形式的query
    print(f"text:{text}")

    if CG_relations == None:
        _, CG_relations = get_entitise_relations(cgdata)
    else:
        CG_relations = list(CG_relations)
    # CG_entities, CG_relations = get_entitise_relations(cgdata)
        
    print(CG_relations)
    
    # TopicEntityName2ID
    if TopicEntityID!= None:
        text[0] = re.sub(r'head_entity=\'.*?\'', f'head_entity=\'{TopicEntityID[2:-2]}\'', text[0])
        text[0] = re.sub(r'tail_entity=\'.*?\'', f'tail_entity=\'{TopicEntityID[2:-2]}\'', text[0])

    Re_entities, Re_relations = get_parameters(text) #参数提取
    print(f"Re_entities:{Re_entities}")
    print(f"Re_relations:{Re_relations}")
    
    # retrieve relations
    if Re_relations == []:
        label_rel = []
    else:
        retriever = OpenAIRetriever(CG_relations)
        top1_api = retriever.get_topk_candidates(1,Re_relations)
        label_rel = [CG_relations[char[0]] for char in top1_api]

    text_query, id_query = replace_query(text, Re_entities, Re_entities, Re_relations, label_rel, cgdata.node2id)
    
    return text_query, id_query, step_text

def TEMP_text2query(args, text, question, temp_data, relation_list, annotation):
    """translate text into formal queries
    output: query class
    """
    pattern_step = r"(Step.*)"
    step_text = re.findall(pattern_step, text) # 得到list形式的query
    print(f"step_text:{step_text}")

    pattern_query = r'Query\d+\s*:\s*"(.*?)"(\n|$)'
    tmp_text = re.findall(pattern_query, text) # 得到list形式的query
    text = []
    for item in tmp_text:
        text.append(item[0].replace('"', "'"))
    print(f"text:{text}")

    # for item in text: item = item.replace("\"", "'")

    CG_relations = list(relation_list)
    CG_relations.append('time')
    CG_relations.append('start_time')
    CG_relations.append('end_time')
    if type(annotation) != dict:
        # 将annotation按, 分割为两个部分
        # 匹配 [] 部分
        res_entity_list = re.search(r'\[.*?\]', annotation).group(0)
        # 匹配 {} 部分
        annotation = re.search(r'\{.*?\}', annotation).group(0)
        
        res_entity_list = ast.literal_eval(res_entity_list)
        annotation = ast.literal_eval(annotation)
        CG_entities = [item for key,item in annotation.items() ]
        CG_entities.extend(res_entity_list)
    else:
        CG_entities = [item for key,item in annotation.items() ]
    
    # CG_entities = [item for key,item in annotation.items() ]
    # CG_entities.extend(res_entity_list)

        
    print(f"CG_relations:{CG_relations}")
    print(f"CG_entities:{CG_entities}")

    Re_entities, Re_relations = get_parameters(text) #参数提取
    print(f"Re_entities:{Re_entities}")
    print(f"Re_relations:{Re_relations}")
    
    # retrieve relations
    if Re_relations == []:
        label_rel = []
    else:
        retriever = OpenAIRetriever(CG_relations)
        top1_api = retriever.get_topk_candidates(1,Re_relations)
        label_rel = [CG_relations[char[0]] for char in top1_api]

    text_query, id_query = replace_query(text, Re_entities, Re_entities, Re_relations, label_rel, temp_data.node2id)
    
    return text_query, id_query, step_text

def get_parameters4TKG(responce_from_api):

    # def has_single_quote_in_double_quotes(s):
    #     # 匹配一对双引号中的内容
    #     matches = re.findall(r'"(.*?)"', s)
    #     # 判断每一对双引号中的内容是否包含单引号
    #     return any("'" in match for match in matches)
    entities = []
    relations = []
    # pattern = r"'(.*?)'"
    rel_pattern = r"(?:relation|key)\s*=\s*'(.*?)'(?:,|\))"
    ent_pattern = r"(?:entity|value)\s*=\s*'(.*?)'(?:,|\))"

    for item in responce_from_api:
        item = item.replace("\"", "'")
        matches_ent = re.findall(ent_pattern, item)
        matches_rel = re.findall(rel_pattern, item)
        for match in matches_ent:
            if match[:6] != 'output' and getStr(match) not in entities and getStr(match) != '':
                entities.append(getStr(match))
        for match in matches_rel:
            if match[:6] != 'output' and getStr(match) not in relations and getStr(match) != '':
                relations.append(getStr(match))

    return entities, relations