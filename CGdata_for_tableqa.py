import json 
from tqdm import tqdm
import argparse
import os
import structllm as sllm
from collections import defaultdict
import random
import multiprocessing as mp
import sys

def TableID_Question_Answer(args, all_data, idx, api_key, table_data):
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
                        tmp_result = sllm.tableqa.tableqa(args, question, table_data[table_id])
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
                    tmp_result = sllm.tableqa.tableqa(args, question, table_data[table_id])
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

def read_csv(csv_file):
    with open(csv_file, 'r', )as f:
        table=[]
        for line in f.readlines():
            table.append(line.strip().split('\t'))
            # print(line, len(table[-1]))
    try:
        assert len(set([len(i) for i in table]))==1
    except Exception as e:
        raise Exception('fail to read csv file, different lengths of rows')
    
    return table

def csv2CG(csv_file):
    PAD = '[0]'
    triples_cg = set()
    entities_2_line = defaultdict(set)    
    table = read_csv(csv_file)
    all_lines_id = set()
    cols = table[0] + ["row_number"]# 增加一列 row_number
    for idx, rows in enumerate(table[1:]):
        row_number = str(idx+1)
        vals = rows + [row_number]
        key = f'[line_{row_number}]'# 人工添加主键 头实体
        all_lines_id.add(key)
        for c,v in zip(cols,vals):
            triples_cg.add((key, c, PAD))
            triples_cg.add((c, v, key))
            entities_2_line[(v, c)].add(key)
            
    entities = list(entities_2_line.keys())    
    return {
        'table_id': '1',
        'triples': triples_cg,
        'entities': entities,  #2-n行所有尾实体
        'entities_2_line': entities_2_line,
        'all_lines_id': all_lines_id, 
        'relations': cols, #第一行
    }

def read_table_data(args):
    print('read table data...')
    table_data = dict()
    file_names = os.listdir(args.folder_path) # 使用os.listdir()函数获取文件夹下所有文件和子文件夹的名称

    paths_of_files = [] # 所有csv文件路径
    for file_name in file_names:
        path_of_file = os.path.join(args.folder_path, file_name)
        if not os.path.isdir(path_of_file):
            paths_of_files.append(path_of_file)
        else:
            for child_file in os.listdir(path_of_file):
                paths_of_files.append(os.path.join(path_of_file, child_file))
        
    error_list = []
    with open("error_file.txt", "w") as f:
        for path_of_file in tqdm(paths_of_files):
            if path_of_file.count("/") != 4: 
                table_name = path_of_file.split('/')[-1].split('.')[0]  # 最后一个/和.之间的字符串
            else: 
                table_name = path_of_file[path_of_file.find('/', path_of_file.find('/') + 1) + 1:].split('.')[0] # 第二个/后的字符串
            
            try:
                test_table_ = csv2CG(path_of_file)
                table_data[table_name] = sllm.cg.data(test_table_['triples'], test_table_['entities_2_line'], test_table_['all_lines_id'])
            except:
                f.write(table_name)
                error_list.append(table_name)
                f.write("\n")
    
    questionTODOset = set()

    with open('output/wikisql/V1_gpt4_f/all_result_2.txt', 'r') as f:
        timeout_data = f.readlines()
        for line in timeout_data:
            line_dict = json.loads(line)
            for key in line_dict.keys():
                questionTODOset.add(line_dict[key]['question'])
    
    print(f"len: {len(questionTODOset)}")
    with open(args.data_path, 'r') as fp:
        tb_question = json.loads(fp.read())

    tb_question = {k.split('.')[0]: v for k, v in tb_question.items() if k.split('.')[0] not in error_list}

    TableQA_data = []
    for table_id in tb_question.keys():
        for qa in tb_question[table_id]:
            question = qa[0]
            if question in questionTODOset:
                continue
            answer = qa[1]
            TableQA_data.append((table_id, question, answer))
    
    return table_data, TableQA_data

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
    table_data, TableQA_data = read_table_data(args) # get CGdata and QAdata
    TableQA_data = TableQA_data[:10]

    if args.num_process == 1:
        TableID_Question_Answer(args, TableQA_data, -1, args.key, table_data)
    else:
        num_each_split = int(len(TableQA_data) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(TableQA_data))
            else:
                end = (idx + 1) * num_each_split
            split_data = TableQA_data[start:end]
            p.apply_async(TableID_Question_Answer, args=(args, split_data, idx, all_keys[idx], table_data))
        p.close()
        p.join()
        print("All of the child processes over!")

        
