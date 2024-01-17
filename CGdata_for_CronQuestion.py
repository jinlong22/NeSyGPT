import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
import multiprocessing as mp
import sys
from collections import defaultdict
import pandas as pd

def TEMP_Question_Answer(args, all_data, idx, api_key, table_data, relations = None):
    # openai.api_key = api_key
    args.key = api_key
    # print(f"args.key:{args.key}, idx:{idx}")

    if idx == -1:
        output_detail_path = args.output_detail_path
        output_result_path = args.output_result_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_detail_path = args.output_detail_path + "_" + idx
        output_result_path = args.output_result_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), output_result_path))

    gpt4prompt_list = [] # 用于存储gpt4的prompt

    with open(output_result_path+".txt", "w") as fresult:
        with open(output_detail_path+".txt", "w") as fdetail:
            # for (table_id, question, answer) in tqdm(all_data, total=len(all_data), desc="PID: %d" % os.getpid()):
            for (question, relation_list, annotation, answer, answer_type, _type) in tqdm(all_data, total=len(all_data), desc="PID: %s" % idx):
                fdetail.write(f"********* Table_temp *********\n")
                fdetail.write(f"=== Question:{question}\n")
                fdetail.write(f"=== Answer:{answer}\n")
                
                if not args.debug:
                    try:
                        sys.stdout = fdetail
                        tmp_result = sllm.tempqa.tempqa(args, question, table_data, relation_list, annotation, gpt4prompt_list)
                        sys.stdout = sys.__stdout__  # 恢复标准输出流
                        fdetail.write(f"=== Answer:{answer}\n")
                        fdetail.write(f"=== Result:{tmp_result}\n")
                        result = [ list(sample) if type(sample)==set else sample for sample in tmp_result ]
                        print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}, answer_type:{answer_type}, type:{_type}")
                        result_dict = dict()
                        tmp_dict = {"question":question,"label":answer,"prediction":result,"answer_type":answer_type,"type":_type}
                        result_dict = tmp_dict
                        fresult.write(json.dumps(result_dict) + "\n")
                        fdetail.write(json.dumps(result_dict) + "\n")
                        fdetail.flush()
                        fresult.flush()

                    except Exception as e:    
                        tmp_dict = {"question":question, "answer": answer, "error": str(e)}
                        if args.store_error:
                            error_path = os.path.join(output_detail_path[:output_detail_path.rfind("/")], args.error_file_path)
                            with open(error_path, "a") as f:
                                f.write(json.dumps(tmp_dict) + "\n")

                else:
                    # sys.stdout = fdetail
                    tmp_result = sllm.tempqa.tempqa(args, question, table_data, relation_list, annotation, gpt4prompt_list)
                    # sys.stdout = sys.__stdout__  # 恢复标准输出流
                    fdetail.write(f"=== Answer:{answer}\n")
                    fdetail.write(f"=== Result:{tmp_result}\n")
                    result = [ list(sample) if type(sample)==set else sample for sample in tmp_result ]
                    print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}, answer_type:{answer_type}, type:{_type}")
                    result_dict = dict()
                    tmp_dict = {"question":question,"label":answer,"prediction":result,"answer_type":answer_type,"type":_type}
                    result_dict = tmp_dict
                    fresult.write(json.dumps(result_dict) + "\n")
                    fdetail.write(json.dumps(result_dict) + "\n")
                    fdetail.flush()
                    fresult.flush()
    
    # dict_gpt4prompt_list = {"prompt":gpt4prompt_list}
    df1 = pd.DataFrame({'prompt':gpt4prompt_list})
    df1.to_csv('TempKG_prompts.csv',index=False)
    print(len(gpt4prompt_list))

def temp2CG(tempkg_file):
    PAD = '[0]'
    triples_cg = set() 
    relations = set()
    entities_2_line = defaultdict(set)
    all_lines_id = set()
    
    with open(tempkg_file, 'r', )as f:
        table=[]
        for idx, line in enumerate(f.readlines()):                        
            elements = line.strip().split('\t')
            try:
                assert len(elements) == 5                
            except Exception as e:
                raise Exception(f'Fail to read {tempkg_file}, elements in row{idx+1} != 5: {line}')
            
            h, r, t, start_, end_ = elements
            triples_cg.add((h, r, PAD))
            triples_cg.add((r, t, h))
            entities_2_line[(t, r)].add(h)#尾实体+关系 对应的头实体(行) 有哪些
            
            triples_cg.add(('start_time', start_, (h,r,t)))
            triples_cg.add(((h,r,t), 'start_time', PAD))
            triples_cg.add(('end_time', end_, (h,r,t)))
            triples_cg.add(((h,r,t), 'end_time', PAD))
            entities_2_line[(start_, 'start_time')].add((h,r,t))
            entities_2_line[(end_, 'end_time')].add((h,r,t))
            
            for e_time in range(int(start_), int(end_)+1):
                triples_cg.add(('time', str(e_time), (h,r,t)))
                triples_cg.add(((h,r,t), 'time', PAD))
                entities_2_line[(str(e_time), 'time')].add((h,r,t))
            
            all_lines_id.add(h)
            all_lines_id.add((h,r,t))
            relations.add(r)
            relations.add('time')
            relations.add('start_time')
            relations.add('end_time')
            
    entities = list(entities_2_line.keys())
    return {
        'table_id': '1',
        'triples': triples_cg,
        'entities': entities,
        'entities_2_line': entities_2_line,
        'all_lines_id': all_lines_id, 
        'relations': list(relations),
    }

def _temp2CG_(args):
    print('read data...')
    test_table_ = temp2CG(args.folder_path)
    KG_data = sllm.cg.data(test_table_['triples'], test_table_['entities_2_line'], test_table_['all_lines_id'], if_temp=True)
    
    # get question, answer
    with open(args.data_path, 'r') as fp:
        tb_question = json.loads(fp.read())

    # question_list = []
    # with open('output/temp/V6_all_v4prompt/all_result.txt', 'r') as fp:
    #     for line in fp.readlines():
    #         question_list.append(json.loads(line)['question'])
        

    KGQA_data = []
    for KG_id in tb_question.keys():
        question = tb_question[KG_id]['question']
        answer = tb_question[KG_id]['answer'][1]
        relations = tb_question[KG_id]['relations']
        
        answer_type = tb_question[KG_id]['answer_type']
        _type = tb_question[KG_id]['type']

        id2entity = tb_question[KG_id]['entities']
        annotation = tb_question[KG_id]['annotation']

        resEntity = [item for entity_key, item in id2entity.items() if entity_key not in annotation.values() ]
        
        if resEntity != []:
            # import pdb; pdb.set_trace()
            if len(resEntity) > 1 or '{tail2}' not in question:
                assert False, (resEntity, question)
                
            # 用resEntity中的第一个元素替换question中的{tail2}
            question = question.replace('{tail2}', resEntity[0])
            
            # print(f"KG_id:{KG_id}, question:{question}, resEntity:{resEntity}")

        # if _type != 'time_join' and _type != 'before_after' and _type != 'first_last':
        #     continue
        
        # if _type != 'first_last' or '\'s' in question or '\'' not in question:
        #     continue
        
        # if question in question_list:
        #     continue

        # if _type != 'before_after':
        #     continue

        relation_list = [item for key,item in relations.items()]

        for tmp_key in annotation.keys():
            if annotation[tmp_key][0] == 'Q' and annotation[tmp_key][1:].isdigit():
                annotation[tmp_key] = id2entity[annotation[tmp_key]]

        if resEntity == []:
            KGQA_data.append((question, relation_list, annotation, answer, answer_type, _type))
        else:
            tmp_annotation = str(annotation)+", "+str(resEntity)
            KGQA_data.append((question, relation_list, tmp_annotation, answer, answer_type, _type))
        # print(f"question:{question}\n, relation_list:{relation_list}\n, annotation:{annotation}\n, answer:{answer}\n, answer_type:{answer_type}\n, type:{type}\n")
        # import pdb; pdb.set_trace()

    return KG_data, KGQA_data

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--folder_path', default="dataset/WikiSQL_TB_csv/test", type=str, help='The CSV data pth.')
    parser.add_argument('--data_path', default="dataset/WikiSQL_CG", type=str, help='The CG data pth.')
    parser.add_argument('--prompt_path', default="structllm/prompt_/wikisql.json", type=str, help='The prompt pth.')
    parser.add_argument('--model', default="gpt-3.5-turbo", type=str, help='The openai model. "gpt-3.5-turbo" and "text-davinci-003" are supported')
    parser.add_argument('--key', default="", type=str, help='The key of openai or path of keys')
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')
    parser.add_argument('--add_retrieve', default=False, type=bool, help='add node retrieve and add retrieved results to prompt')
    parser.add_argument('--output_detail_path', default="output/V3/output_detail", type=str)
    parser.add_argument('--output_result_path', default="output/V3/output_result", type=str)
    parser.add_argument('--error_file_path', default="timeout_file.txt", type=str)
    parser.add_argument('--prompt_csv_path', default="prompt4TempQA/TempKG_prompts_1.0_response.csv", type=str)
    parser.add_argument('--start_question_idx', default=1, type=int, help='start_question_idx split from 1500')
    parser.add_argument('--store_error', action="store_true", default=True)
    parser.add_argument('--SC_Num', default=5, type=int)
    parser.add_argument('--debug', default=0, type=int)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    print(f"SC_Num:{args.SC_Num}\n")
    # get API key
    if not args.key.startswith("sk-"):
        with open(args.key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
   
    # get data
    KG_data, KGQA_data = _temp2CG_(args) # get CGdata and QAdata

    if args.num_process == 1:
        TEMP_Question_Answer(args, KGQA_data, -1, args.key, KG_data)
    else:
        num_each_split = int(len(KGQA_data) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(KGQA_data))
            else:
                end = (idx + 1) * num_each_split
            split_data = KGQA_data[start:end]
            p.apply_async(TEMP_Question_Answer, args=(args, split_data, idx, all_keys[idx], KG_data))
        p.close()
        p.join()
        print("All of the child processes over!")