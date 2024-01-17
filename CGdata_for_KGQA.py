import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
import random
import multiprocessing as mp
import sys
from collections import defaultdict

def KGID_Question_Answer(args, all_data, idx, api_key, table_data, relations):
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

    with open(output_result_path+".txt", "w") as fresult:
        with open(output_detail_path+".txt", "w") as fdetail:
            # for (table_id, question, answer) in tqdm(all_data, total=len(all_data), desc="PID: %d" % os.getpid()):
            for (table_id, question, answer) in tqdm(all_data, total=len(all_data), desc="PID: %s" % idx):
                fdetail.write(f"********* Table{table_id} *********\n")
                fdetail.write(f"=== Question:{question}\n")
                fdetail.write(f"=== Answer:{answer}\n")
                
                if not args.debug:
                    try:
                        sys.stdout = fdetail
                        tmp_result = sllm.kgqa.kgqa(args, question, table_data[table_id], relations)
                        sys.stdout = sys.__stdout__  # 恢复标准输出流
                        fdetail.write(f"=== Answer:{answer}\n")
                        fdetail.write(f"=== Result:{tmp_result}\n")
                        result = [ list(sample) if type(sample)==set else sample for sample in tmp_result ]
                        print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}")
                        result_dict = dict()
                        tmp_dict = {"question":question,"label":answer,"prediction":result}
                        result_dict[table_id] = tmp_dict
                        fresult.write(json.dumps(result_dict) + "\n")
                        fdetail.write(json.dumps(result_dict) + "\n")
                        fdetail.flush()
                        fresult.flush()

                    except Exception as e:    
                        tmp_dict = {"tableid":table_id, "question":question, "answer": answer, "error": str(e)}
                        if args.store_error:
                            error_path = os.path.join(output_detail_path[:output_detail_path.rfind("/")], args.error_file_path)
                            with open(error_path, "a") as f:
                                f.write(json.dumps(tmp_dict) + "\n")

                else:
                    # sys.stdout = fdetail
                    tmp_result = sllm.kgqa.kgqa(args, question, table_data[table_id], relations)
                    # sys.stdout = sys.__stdout__  # 恢复标准输出流
                    fdetail.write(f"=== Answer:{answer}\n")
                    fdetail.write(f"=== Result:{tmp_result}\n")
                    result = [ list(sample) if type(sample)==set else sample for sample in tmp_result ]
                    print(f"label:{answer}, result:{result}, output_result_path:{output_result_path}")
                    result_dict = dict()
                    tmp_dict = {"question":question,"label":answer,"prediction":result}
                    result_dict[table_id] = tmp_dict
                    fresult.write(json.dumps(result_dict) + "\n")
                    fdetail.write(json.dumps(result_dict) + "\n")
                    fdetail.flush()
                    fresult.flush()

def kg2CG(args):

    print('read table data...')
    KG_data = dict()
    kg_file = args.folder_path

    # get KG data
    PAD = '[0]'
    triples_cg = set() 
    relations = set()
    entities_2_line = defaultdict(set)
    all_lines_id = set()
    
    with open(kg_file, 'r', )as f:
        for idx, line in enumerate(f.readlines()):                        
            elements = line.strip().split('\t')
            try:
                assert len(elements) == 3                
            except Exception as e:
                raise Exception(f'Fail to read {kg_file}, elements in row{idx+1} != 3: {line}')
            
            h, r, t = elements
            triples_cg.add((h, r, PAD))
            triples_cg.add((r, t, h))
            entities_2_line[(t, r)].add(h)#尾实体+关系 对应的头实体(行) 有哪些
            all_lines_id.add(h)
            relations.add(r)
    
    KG_name = 'main'
    KG_data[KG_name] = sllm.cg.data(triples_cg, entities_2_line, all_lines_id)

    # entities = list(entities_2_line.keys())

    # get question, answer
    with open(args.data_path, 'r') as fp:
        tb_question = json.loads(fp.read())

    # tb_question = {k.split('.')[0]: v for k, v in tb_question.items()}

    KGQA_data = []
    for KG_id in tb_question.keys(): # 这个
        question = tb_question[KG_id]['question']
        answer = tb_question[KG_id]['answer']
        KGQA_data.append((KG_name, question, answer))

    return KG_data, KGQA_data, relations


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
    KG_data, KGQA_data, relations = kg2CG(args) # get CGdata and QAdata

    if args.num_process == 1:
        KGID_Question_Answer(args, KGQA_data, -1, args.key, KG_data, relations)
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
            p.apply_async(KGID_Question_Answer, args=(args, split_data, idx, all_keys[idx], KG_data, relations))
        p.close()
        p.join()
        print("All of the child processes over!")

        
