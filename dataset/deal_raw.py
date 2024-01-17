from tqdm import tqdm
import json
from collections import defaultdict
import os
import pandas as pd
import shutil
import pickle
import numpy as np
import time 
from glob import glob
import re 

def deal_temp():
    input_qa_file = 'raw/temporal/data/wikidata_big/questions/valid.pickle'
    # aliases_file = 'raw/temporal/data/wikidata_big/kg/wd_id_to_aliases.pickle'
    qa_json_out = 'temp_CG/qa_valid.jsonl'
    kg_file = 'raw/temporal/data/wikidata_big/kg/full.txt'# Q311641没有在test中
    output_kg_file = 'temp_CG/kg.txt'
    entity_to_name_file  = 'raw/temporal/data/wikidata_big/kg/wd_id2entity_text.txt'
    rel_to_name_file  = 'raw/temporal/data/wikidata_big/kg/wd_id2relation_text.txt'
    qa_data = pickle.load(open(input_qa_file, 'rb')) 
    # aliases_data = pickle.load(open(aliases_file, 'rb'))
    # import pdb; pdb.set_trace()
    large_qa_dict = {}
    ent_names = {}
    rel_names = {}
    with open(entity_to_name_file, 'r') as f:
        for line in f.readlines():
            id_ = line.strip().split()[0]
            name_ = ' '.join(line.strip().split()[1:])
            ent_names[id_] = name_
    with open(rel_to_name_file, 'r') as f:
        for line in f.readlines():
            id_, name_ = line.strip().split('\t')
            rel_names[id_] = name_

    for idx_, line in enumerate(qa_data):
        qa_dict_ = {}
        qa_dict_['question_raw'] = line['question']
        que_str = line['question']
        need_trans = re.findall(r'Q\d+', que_str)
        if len(need_trans)==0:
            5/0
        else:
            for i_ in need_trans:
                que_str = que_str.replace(i_, ent_names[i_])
        #import pdb; pdb.set_trace()
        qa_dict_['question'] = que_str
        qa_dict_['template'] = line['template']
        qa_dict_['type'] = line['type']
        qa_dict_['times'] = list(line['times'])
        qa_dict_['annotation'] = line['annotation']
        qa_dict_['paraphrases'] = line['paraphrases']
        qa_dict_['entities'] = {}
        qa_dict_['relations'] = {}
        qa_dict_['uniq_id'] = line['uniq_id']
        for ent_id in line['entities']:
            qa_dict_['entities'][ent_id] = ent_names[ent_id]
        for rel_id in line['relations']:
            qa_dict_['relations'][rel_id] = rel_names[rel_id]
        qa_dict_['answer_type'] = line['answer_type']
        qa_dict_['answer'] = ([],[])
        for ans_id in line['answers']:
            qa_dict_['answer'][0].append(ans_id)
            if ans_id in ent_names:
                qa_dict_['answer'][1].append(ent_names[ans_id])
            else:
                qa_dict_['answer'][1].append(ans_id)

        large_qa_dict[idx_] = qa_dict_

    with open(qa_json_out, 'w') as fp:
        json.dump(large_qa_dict, fp, ensure_ascii=False)
    with open(kg_file, 'r') as f:
        with open(output_kg_file,'w') as f_o:
            for line in f.readlines():
                h, r, t, st, et = line.strip().split()
                h = ent_names[h]
                r = rel_names[r]
                t = ent_names[t]
                f_o.write('\t'.join([h,r,t,st,et])+'\n')

    # import pdb; pdb.set_trace()

def deal_wqsp():
    input_qa_file = 'raw/WQSP/webqsp_simple_test.jsonl'
    qa_json_out = 'WQSP_CG/qa_test.jsonl'    
    ent2idfile = 'raw/WQSP/ent2id.pickle'
    rel2idfile = 'raw/WQSP/rel2id.pickle'
    entnameflie = 'raw/WQSP/entity_name.pickle'
    ent_type_file = 'raw/WQSP/ent_type_ary.npy'
    subgraph_file = 'raw/WQSP/subgraph_2hop_triples.npy'
    onehop_graph_file = f'raw/WQSP/1hop-sub.pickle'
    
    with open(ent2idfile, 'rb') as f:
        print(f'step1: 加载{ent2idfile}')
        ent2id = pickle.load(f) # 34615716 entities
    # ent2type = np.load(ent_type_file) # 34615716 entities
    
    topic_ents = set()
    topic_ents_2hop = set()
    topic_fb_ids_1hop = set()
    topic_fb_ids_2hop = set()
    qa_dict_ = {}
    def read_name_info():
        with open(entnameflie, 'rb') as f:
            print(f'加载{entnameflie}')
            ent2name = pickle.load(f) # [0] 34615510   [1] 12243748
        return ent2name
    ent2name = read_name_info()

    with open(input_qa_file, 'r') as f:
        for idx_, line in enumerate(f.readlines()):
            thisq = json.loads(line.strip())
            ent_fb_id=thisq['TopicEntityID']
            _ent_name = ent2name[1][ent_fb_id]
            qa_dict_[idx_] = {'question':thisq["Question"], 'answer':thisq["Answers"], "TopicEntityName": _ent_name, "TopicEntityID": thisq['TopicEntityID']}
            topic_fb_ids_1hop.add(ent_fb_id)
            topic_ents.add(ent2id[ent_fb_id])#都是一跳的

        with open(qa_json_out, 'w') as fp:
            json.dump(qa_dict_, fp, ensure_ascii=False)

    def read_fb_id_info():
        print(f"ent2id 转换get id2ent")
        id2ent = {value:key for key, value in ent2id.items()}
        with open(rel2idfile, 'rb') as f:
            print(f'加载{rel2idfile}')
            rel2id = pickle.load(f) # 15964 relations
            id2rel = {value:key for key, value in rel2id.items()}
        return id2ent, id2rel
    
    
    
    def triple2string_fb(hop, topic_fb_ids, id2ent, id2rel):#转换成需要的格式
        out_dir = f'raw/WQSP/subgraph{hop}hop-fb/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        with open(f'raw/WQSP/{hop}hop-sub.pickle', 'rb') as f_in:
            print(f'加载{hop}跳子图 ...')
            ent2triples = pickle.load(f_in)

        for tar_ent_fb in topic_fb_ids:
            entid = ent2id[tar_ent_fb]
            the_graph_file = out_dir + tar_ent_fb + '.txt'
            with open(the_graph_file, 'w') as f_out:
                for h,r,t in ent2triples[entid]:
                    h_id_fb = id2ent[h]
                    t_id_fb = id2ent[t]
                    r_name = id2rel[r]
                    f_out.write('\t'.join([h_id_fb, r_name, t_id_fb])+'\n')

    
    def triple2string_name(hop, topic_fb_ids, ent2name):#转换成需要的格式
        def find_name(ent):
            try:
                ent_name = ent2name[1][ent]
            except:#没有的话
                ent_name = ent
                #if not ent.startswith('m.') and not ent.startswith('g.'):
                 #   print(ent_name)
            return ent_name
        
        read_dir = f'raw/WQSP/subgraph{hop}hop-fb/'
        out_dir = f'WQSP_CG/subgraph{hop}hop/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for tar_ent_fb in topic_fb_ids:
            the_graph_file = out_dir + tar_ent_fb + '.txt'
            the_graph_relation_file = out_dir + tar_ent_fb + '(relation).txt'
            read_graph_file = read_dir + tar_ent_fb + '.txt'
            r_set = set()
            with open(read_graph_file, 'r') as f_in:
                with open(the_graph_file, 'w') as f_out:
                    for line in f_in.readlines():
                        h,r,t = line.strip().split('\t')
                        h_name = find_name(h)
                        r_set.add(r)
                        t_name = find_name(t)
                        f_out.write('\t'.join([h_name, r, t_name])+'\n')
            with open(the_graph_relation_file, 'w') as f_o:
                print(tar_ent_fb, len(r_set))
                for e_r in r_set:
                    f_o.write(e_r+'\n')
        import pdb; pdb.set_trace()

    def extract_onehop_(seeds_ents):
        seed_ent2triple = defaultdict(set)        
        subgraph = np.load(subgraph_file) # 105948364 triples
        for h,r,t in tqdm(subgraph):#全图
            if h in seeds_ents:#只以seeds_ents中的为主键
                seed_ent2triple[h].add((h,r,t))
            if t in seeds_ents:
                seed_ent2triple[t].add((h,r,t))
            
        if os.path.exists(onehop_graph_file):
            with open(onehop_graph_file, 'rb') as f_pkl:
                print(f'加载已有一跳子图{onehop_graph_file}...')
                old_seed2triple = pickle.load(f_pkl)
            seed_ent2triple.update(old_seed2triple)
        with open(onehop_graph_file, 'wb') as f_pkl:
            print(f'更新一跳子图{onehop_graph_file}...')
            pickle.dump(seed_ent2triple, f_pkl)
        print(f'已更新文件{onehop_graph_file}')
        return seed_ent2triple
    
    def extract_morehop(target_hop, target_ents):
        last_hop_graph_file = f'raw/WQSP/{target_hop-1}hop-sub.pickle'
        this_hop_graph_file = f'raw/WQSP/{target_hop}hop-sub.pickle'
        
        with open(onehop_graph_file, 'rb') as fin:
            print(f'加载已有一跳子图{onehop_graph_file}...')
            onehop_sub = pickle.load(fin)

        with open(last_hop_graph_file, 'rb') as f_in:
            if last_hop_graph_file==onehop_graph_file:
                last_hop_sub = onehop_sub#避免加载两次，可以复用的
            else:
                print(f'加载上一跳数据{last_hop_graph_file}')
                last_hop_sub = pickle.load(f_in)
        
        this_hop_sub = defaultdict(set)

        # n-1 hop相关实体作为seeds entity, 找1hop邻居（基础扩展）
        related_ents = set()
        for tar_ent in target_ents:
            for h,r,t in last_hop_sub[tar_ent]:
                related_ents.add(h)
                related_ents.add(t)                

        has_one_hop = set(list(onehop_sub.keys()))
        new_seed_ent = related_ents - has_one_hop
        print(f'相关实体{len(related_ents)}个，已有1hop记录的实体{len(has_one_hop)}个')
        
        if len(new_seed_ent)>0:
            print(f'为新增{len(new_seed_ent)}个实体寻找一阶邻居')
            onehop_sub = extract_onehop_(seeds_ents=new_seed_ent)
            # 完成基础扩展之后，更新one_graph_file
        else:
            print(f'无需扩展one hop邻居')

        for tar_ent in target_ents:
            import pdb; pdb.set_trace()
            this_hop_sub[tar_ent] = set([i for i in last_hop_sub[tar_ent]])#复制
            last_cnt = len(this_hop_sub[tar_ent])
            for h,r,t in last_hop_sub[tar_ent]:
                this_hop_sub[tar_ent] |= onehop_sub[h]# 找到h的一跳子图，融合
                this_hop_sub[tar_ent] |= onehop_sub[t]# 找到t的一跳子图，融合 
                print(f"根据({h},{r},{t})，新增三元组{len(this_hop_sub[tar_ent])-last_cnt}")
                last_cnt = len(this_hop_sub[tar_ent])
            print(f'{tar_ent}的三元组{len(this_hop_sub[tar_ent])}个，last hop = {len(last_hop_sub[tar_ent])}个')

        if os.path.exists(this_hop_graph_file):
            with open(this_hop_graph_file, 'rb') as f_pkl:
                print(f'加载已有{target_hop}跳子图 from {this_hop_graph_file}')
                old_ent2triples = pickle.load(f_pkl)
            this_hop_sub.update(old_ent2triples)
        with open(this_hop_graph_file, 'wb') as f_pkl:
            print(f'更新{target_hop}跳子图 save to {this_hop_graph_file}')
            pickle.dump(this_hop_sub, f_pkl)
        print(f'已更新文件{this_hop_graph_file}')
        return this_hop_sub


    # -----------------------------------------
    # 1-hop # 一次性抽取所有seed ent 的一跳1hop 需遍历全图
    # -----------------------------------------
    # extract_onehop_(seeds_ents=topic_ents)

    # -----------------------------------------
    # more-hop # 在n-1 hop的基础上，以n-1 hop相关实体作为seeds entity, 找1hop邻居（遍历全图），再合并
    # -----------------------------------------
    # extract_morehop(2, target_ents=topic_ents_2hop)

    # -----------------------------------------
    # 数字id转换成 fb_id 写入文件 
    # -----------------------------------------
    # id2ent, id2rel = read_fb_id_info()
    # triple2string_fb(1, topic_fb_ids_1hop, id2ent, id2rel)
    # triple2string_fb(2, topic_fb_ids_2hop, id2ent, id2rel)

    # -----------------------------------------
    # fb_id转换成 name，不用访问subgraph.pickle 
    # -----------------------------------------
    triple2string_name(1, topic_fb_ids_1hop, ent2name)
    # triple2string_name(2, topic_fb_ids_2hop, ent2name)

def deal_wtq():
    tb_dir = 'WTQ/csv/'
    if True: # for train_question
        qa_file = 'raw/WikiTableQuestions/data/training.tsv'
        raw_tb_dir = 'raw/WikiTableQuestions/csv/'
        qa_json_out = 'WTQ/train.jsonl'
        qa_dict = defaultdict(list)
        with open(qa_file, 'r') as f:
            for line in tqdm(f):
                eles = line.strip().split('\t')
                ques_, table_id = eles[1], eles[2]
                ans_ = eles[3:]
                _, sub_dir, tbname = table_id.split('/')
                target_tb_name = tb_dir+sub_dir+'/'+tbname
                if not os.path.exists(tb_dir+sub_dir):
                    os.mkdir(tb_dir+sub_dir)
                if not os.path.exists(target_tb_name):
                    try:
                        raw_tb_name = (raw_tb_dir+sub_dir+'/'+tbname).replace('.csv','.tsv')
                        each_line_len=set()
                        with open(raw_tb_name) as f_in:
                            for line in f_in.readlines():
                                each_line_len.add(len(line.split('\t')))
                        assert len(each_line_len)==1
                        shutil.copy2(raw_tb_name, target_tb_name)
                    except:
                        print(f'{raw_tb_name}原格式错误')
                        import pdb; pdb.set_trace()
                        
                if os.path.exists(target_tb_name):
                    qa_dict[table_id].append([ques_, ans_])  
        
        with open(qa_json_out, 'w') as fp:
            json.dump(qa_dict, fp, ensure_ascii=False)

    if False:# for test_question and tables
        qa_file = 'raw/StructGPT/data/wtq/wikitq_test.json'
        qa_json_out = 'WTQ/test.jsonl'
        qa_dict =  defaultdict(list)
        
        with open(qa_file, 'r') as f_in:
            datas = json.loads(f_in.read())

        for item in datas:
            # dict_keys(['id', 'question', 'table_id', 'table', 'answer_text', 'struct_in', 'text_in', 'seq_out'])
            qa_dict[item['table_id']].append([item['question'], item['answer_text']])
            table = item['table']
            _, sub_dir, tbname = item['table_id'].split('/')
            if not os.path.exists(tb_dir+sub_dir):
                os.mkdir(tb_dir+sub_dir)
            if not os.path.exists(tb_dir+sub_dir+'/'+tbname):
                each_line_len=set()
                with open(tb_dir+sub_dir+'/'+tbname, 'w') as f:                
                    f.write('\t'.join(table['header'])+'\n')
                    each_line_len.add('\t'.join(table['header']).count('\t'))
                    for line in table['rows']:
                        f.write('\t'.join(line)+'\n')
                        each_line_len.add('\t'.join(line).count('\t'))
                if len(each_line_len)>1:
                    import pdb; pdb.set_trace()
                    print(each_line_len)
        with open(qa_json_out, 'w') as fp:
            json.dump(qa_dict, fp)
    
def deal_tabfact():
    qa_file = 'raw/StructGPT/data/tabfact/tab_fact_test.json'
    qa_json_out = 'TabFact/test.jsonl'
    tb_dir = 'TabFact/csv/'
    qa_dict =  defaultdict(list)
    
    with open(qa_file, 'r') as f_in:
        datas = json.loads(f_in.read())
    for item in datas:
        # dict_keys(['id', 'table', 'statement', 'label', 'hardness', 'small_test', 'struct_in', 'text_in', 'seq_out'])
        table = item['table']
        qa_dict[table['id']].append([item['statement'], item['label']])
        
        tbname = table['id']
        if not os.path.exists(tb_dir+tbname):
            each_line_len=set()
            with open(tb_dir+tbname, 'w') as f:                
                f.write('\t'.join(table['header'])+'\n')
                each_line_len.add('\t'.join(table['header']).count('\t'))
                for line in table['rows']:
                    f.write('\t'.join(line)+'\n')
                    each_line_len.add('\t'.join(line).count('\t'))
            if len(each_line_len)>1:
                import pdb; pdb.set_trace()
                print(each_line_len)
    with open(qa_json_out, 'w') as fp:
        json.dump(qa_dict, fp)
    
    
def deal_spider():
    qa_file = 'raw/StructGPT/data/spider/dev.jsonl'
    qa_file_out = f'Spider_CG/qa.jsonl'
    table_file = 'raw/StructGPT/data/spider/all_tables_content.jsonl'
    table_file_out = 'Spider_CG/csv/'
    table_file_merge = 'Spider_CG/csv_merge/'
    
    with open(table_file, 'r') as f_in:
        dbs = json.loads(f_in.read())
    
    for db_name in dbs:
        db_dir = table_file_out+db_name
        big_dfs = {}
        if not os.path.exists(db_dir):
            os.mkdir(db_dir)
        for table_name in dbs[db_name]:
            table_file = db_dir+'/'+table_name+'.csv'
            table_data = dbs[db_name][table_name]
            each_line_len=set()
            if db_name == 'world_1' and table_name=='city':
                table_data['headers'][2] = 'country code'
            data_s = {key:[] for key in table_data['headers']}
            with open(table_file, 'w') as f:
                f.write('\t'.join([str(i).strip() for i in table_data['headers']])+'\n')
                each_line_len.add('\t'.join([str(i).strip() for i in table_data['headers']]).count('\t'))
                for line in table_data['rows']:
                    for ele_idx, ele in enumerate(line):
                        data_s[table_data['headers'][ele_idx]].append(str(ele).strip())
                    f.write('\t'.join([str(i).strip() for i in line])+'\n')
                    each_line_len.add('\t'.join([str(i).strip() for i in line]).count('\t'))
            if len(each_line_len)>1:
                import pdb; pdb.set_trace()
                print(each_line_len)
            
            try:
                df = pd.DataFrame(data_s)  
                # import pdb; pdb.set_trace()
            except:
                import pdb; pdb.set_trace()
            big_dfs[table_name] = df
            # if table_name=='addresses':
            #     big_dfs['current addresses'] = df
            #     big_dfs['permanent addresses'] = df           
                        
                
        # pd.merge(left, right, how='outer', on=['name', 'age'])  # 外连接
        # 找到表的主键和外键
        # 按照主键外键merge
        with open('./raw/Spider/tables.json', 'r') as f_js:
            table_json = json.loads(f_js.read())        
        tabjs={}
        for e_json in table_json:
            tabjs[e_json['db_id']] = e_json
            
        with open(qa_file_out, 'r') as f_read:
            used_dbs = json.loads(f_read.read()).keys()
        if db_name in used_dbs:   
            db_dir_merge = table_file_merge+db_name        
            if not os.path.exists(db_dir_merge):
                os.mkdir(db_dir_merge)
            rest = set(big_dfs.keys())
            # print(rest)
            # for e_res in rest:
                #print('table_name---', e_res)
                #print(big_dfs[e_res])
            
            tab_info = tabjs[db_name]
            col_list = []
            col_to_tb = defaultdict(dict)
            for col_idx, (tb_id, col_name) in enumerate(tab_info['column_names']):
                tabname = tab_info['table_names'][tb_id]
                col_list.append((tabname, col_name))
                
            foreign_key_map = {}# table key: table.key            
            has_child = set()            
            child = set()
            forei_tb = set()
            for k1, k2 in tab_info['foreign_keys']:
                t1,c1 = col_list[k1]
                t2,c2 = col_list[k2]
                foreign_key_map[col_list[k1]] = col_list[k2]                
                has_child.add(t1)
                child.add(t2)
                forei_tb.add(t1)
                forei_tb.add(t2)
            only_father = has_child-child
            only_child = child-has_child
            both_fc = has_child&child
            assert len(only_father|only_child|both_fc) == len(forei_tb)
                
            print(tab_info)
            main_key = {}            
            for k1 in tab_info['primary_keys']:
                main_key[tab_info['table_names'][tab_info['column_names'][k1][0]]] = tab_info['column_names'][k1][1]
                                        
            
#             renames_dict = {'car_1':{'car makers':{'country':'countries.country id'}, 
#                                      'car names':{'model':'model'},
#                                      'countries':{'continent':'continent id'}, 
#                                      'continents':{'cont id':'continent id'},
#                                      'model list':{'maker':'car makers.id','model id':'model id','model':'model'},
#                                      'cars data':{'id':'car names.make id'},
#                                     },
#                             'course_teach':{'course':{},
#                                             'teacher':{},
#                                             'course arrange':{'course id':'course.course id', 'teacher id':'teacher.teacher id'},
#                                            },
#                             'concert_singer':{'singer':{}, 
#                                               'stadium':{},
#                                               'concert':{'stadium id': 'stadium.stadium id'},
#                                               'singer in concert':{'singer id': 'singer.singer id', 'concert id': 'concert.concert id'},
#                                              },
#                             'tvshow':{'tv channel':{}, 
#                                       'tv series':{'channel': 'tv channel.id'}, 
#                                       'cartoon':{'channel': 'tv channel.id'},
#                                      },
#                             'student_transcripts_tracking':{'transcript contents':{'transcript id':'transcripts.transcript id', 'student course id':'student enrolment courses.student course id'}, 
#                                                             'sections':{'course id':'courses.course id'}, 
#                                                             'semesters':{}, 
#                                                             'courses':{}, 
#                                                             'student enrolment':{'semester id':'semesters.semester id','student id':'students.student id','degree program id':'degree programs.degree program id'}, 
#                                                             'transcripts':{}, 
#                                                             'degree programs':{'department id':'departments.department id'}, 
#                                                             'students':{'current address id':'current addresses.address id','permanent address id':'permanent addresses.permanent address id'}, 
#                                                             'departments':{}, 
#                                                             'student enrolment courses':{'course id':'courses.course id','student enrolment id':'student enrolment.student enrolment id'}, 
#                                                             'current addresses':{},
#                                                             'permanent addresses':{}
#                                                            },
                            
                            
#                            }                                                                                   
            
            for tn in big_dfs.keys():
                # renames_dict[db_name][tn].update({i:tn +'.'+ i for i in list(big_dfs[tn].columns) if i not in renames_dict[db_name][tn].keys()})# 更新一下更改列名
                # big_dfs[tn].rename(columns = renames_dict[db_name][tn], inplace = True)
                renames_dict = {i:tn +'.'+ i for i in list(big_dfs[tn].columns)}
                big_dfs[tn].rename(columns = renames_dict, inplace = True)
#             print(big_dfs)
#             df_lst = list(big_dfs.keys())
#             prio = ['course arrange', 'singer in concert', 'concert']
#             if set(prio) & set(df_lst):
#                 key_ = list(set(prio) & set(df_lst))
#                 df_lst = key_ + [i for i in df_lst if i not in key_]
#                 print(f'将key = {key_}提前')
                
#             left = df_lst[0]            
#             left_df = big_dfs[left]
#             for right in df_lst[1:]:
#                 right_df = big_dfs[right]
#                 key = list(set(list(left_df.columns)) & set(list(right_df.columns)))
#                 if len(key)>1:
#                     import pdb; pdb.set_trace()
#                     print('合并过程存在多个键？？')
#                 elif len(key)==0:
#                     import pdb; pdb.set_trace()
#                     print('合并过程无键？？')
#                 left_df = pd.merge(left_df, right_df, how='outer', on=key)  # 外连接 
            # 找到table_next的root作left 没作为值的
            
            for tbkey in foreign_key_map:
                tbkey2 = foreign_key_map[tbkey]
                left_df = big_dfs[tbkey[0]]
                left_key = tbkey[0]+'.'+tbkey[1]
                right_df = big_dfs[tbkey2[0]]
                right_key = tbkey2[0]+'.'+tbkey2[1]
                # if db_name == 'flight_2':
                #     import pdb; pdb.set_trace()
                print(tbkey, tbkey2)
                print(left_df.columns)
                print(right_df.columns)
                big_dfs[tbkey[0]] = pd.merge(left_df, right_df, how='outer', left_on=left_key, right_on=right_key)
                # 如果只作为最后一层，不用改了就
                if tbkey2[0] in has_child:
                    big_dfs[tbkey2[0]] = pd.merge(left_df, right_df, how='outer', left_on=left_key, right_on=right_key)
                # result = pd.merge(left_df, right_df, how='outer', left_on=left_key, right_on=right_key)
            
            # ------- 已经完成扩展，删除冗余表 ---------
            # 此外，还需要把foreign中不涉及的，直接放在大表后面 not in forei_tb
            result_dfs = {}
            # big_dfs里面排个序 
            all_tb_name = list(only_father)+list(both_fc)+list(only_child)+list(set(big_dfs.keys())-forei_tb)
            print(big_dfs.keys())
            print('only_father:', only_father)
            print('both_fc:', both_fc)
            print('only_child:', only_child)
            print('no_foreign:', set(big_dfs.keys())-forei_tb)
            for e_df in all_tb_name:
                if e_df not in forei_tb:
                    result_dfs[e_df] = big_dfs[e_df]
                else:
                    # 1它的所有列被某个已有的完全覆盖，且行数更少/相等：被融入：修改之前的名字 
                    # 2它的所有列将某个已有的完全覆盖，且行数更多：修改之前的名字 和 df
                    # 0其他情况 单独成一个df
                    flag = 0
                    this_cln = set(big_dfs[e_df].columns)
                    for have_df in result_dfs:
                        have_cln = set(result_dfs[have_df].columns)
                        if len(this_cln - have_cln)==0 and len(have_cln - this_cln)>=0:
                            assert len(big_dfs[e_df])<=len(result_dfs[have_df])# 一定被完全覆盖
                            flag = 1
                            result_dfs[have_df+'|'+e_df] = result_dfs[have_df]
                            del result_dfs[have_df]
                            break
                    if flag==0:
                        for have_df in result_dfs:
                            have_cln = set(result_dfs[have_df].columns)
                            if len(this_cln - have_cln)>0 and len(have_cln - this_cln)==0:
                                assert len(big_dfs[e_df])>=len(result_dfs[have_df])
                                flag = 2
                                result_dfs[have_df+'|'+e_df] = big_dfs[e_df]
                                del result_dfs[have_df]
                        if flag ==0:                            
                            # import pdb; pdb.set_trace()
                            result_dfs[e_df] = big_dfs[e_df]
            
            # import pdb; pdb.set_trace()
            merge_names=list(set(result_dfs.keys()))
            print(merge_names)
            # result = result_dfs[merge_names[0]]
            # for e_df in merge_names[1:]:
            #     result = pd.concat([result, result_dfs[e_df]], axis=0, ignore_index=True)
            # print(result)
            for resu_name in merge_names:
                result = result_dfs[resu_name]
                for cln in result.columns:
                    if db_name in ['student_transcripts_tracking']:
                        if cln.startswith('addresses.'):
                            if cln.endswith('_y'):
                                new_cln = cln.replace('_y', '_current')
                                result.rename(columns = {cln:new_cln}, inplace = True)
                            elif cln.endswith('_x'):
                                new_cln = cln.replace('_x', '_permanent')
                                result.rename(columns = {cln:new_cln}, inplace = True)
                    elif db_name in ['flight_2']:
                        if cln.startswith('airports.'):
                            if cln.endswith('_y'):
                                new_cln = cln.replace('_y', '_source')
                                result.rename(columns = {cln:new_cln}, inplace = True)
                            elif cln.endswith('_x'):
                                new_cln = cln.replace('_x', '_destination')
                                result.rename(columns = {cln:new_cln}, inplace = True) 
                    elif db_name == 'wta_1':
                        if cln.startswith('players.'):
                            if cln.endswith('_y'):
                                new_cln = cln.replace('_y', '_loser')
                                result.rename(columns = {cln:new_cln}, inplace = True)
                            elif cln.endswith('_x'):
                                new_cln = cln.replace('_x', '_winner')
                                result.rename(columns = {cln:new_cln}, inplace = True) 
                    elif db_name == 'network_1':
                        if cln.startswith('high schooler.'):
                            if cln.endswith('_y'):
                                if resu_name=='friend':
                                    new_cln = cln.replace('_y', '_student')
                                elif resu_name=='likes':
                                    new_cln = cln.replace('_y', '_liked')
                                result.rename(columns = {cln:new_cln}, inplace = True)
                            elif cln.endswith('_x'):
                                if resu_name=='friend': 
                                    new_cln = cln.replace('_x', '_friend')
                                elif resu_name =='likes':
                                    new_cln = cln.replace('_x', '_student')
                                result.rename(columns = {cln:new_cln}, inplace = True) 
                    else:
                        if cln.endswith('_x'):
                            import pdb; pdb.set_trace()
                            print(cln)
                result.to_csv(db_dir_merge+'/'+resu_name+'.csv' ,index=False)
            # big_csv_file = table_file_out+db_name+'.csv'
            # result.to_csv(big_csv_file, index=False)

        # db = dbs['company_employee']
        # db 的3张小表 dict_keys(['people', 'company', 'employment'])
        # 1张小表 tables['company_employee']['people']
        # {'headers': ['people id', 'age', 'name', 'nationality', 'graduation college'], 'rows': [[1, 27, 'Reggie Lewis', 'United States', 'Northeastern'], [2, 25, 'Brad Lohaus', 'United States', 'Iowa'], [3, 37, 'Tom Sheehey', 'United Kindom', 'Virginia'], [4, 31, 'Darryl Kennedy', 'United States', 'Oklahoma'], [5, 34, 'David Butler', 'United Kindom', 'California'], [6, 37, 'Tim Naegeli', 'United States', 'Wisconsin–Stevens Point'], [7, 30, 'Jerry Corcoran', 'United States', 'Northeastern']]}
    
    
    if not os.path.exists(qa_file_out):
        qa_dict_ = defaultdict(list)
        with open(qa_file, 'r') as f_qa:
            for line in tqdm(f_qa):
                qa = json.loads(line)
                id_ = qa['db_id']
                que = qa['question']
                ans = qa['answer_text']
                qa_dict_[id_].append([que, ans])

                with open(qa_file_out,'w') as fp:
                    json.dump(qa_dict_, fp)
                    
    
        
def deal_metaQA():
    
    input_file = 'raw/MetaQA/kb.txt'
    output_file = 'MetaQA_CG/kg.txt'
    with open(input_file, 'r') as f_in:
        with open(output_file, 'w') as f_out:
            for line in f_in.readlines():
                elements = line.strip().split('|')
                f_out.write('\t'.join(elements)+'\n')
    
    for hop in [1,2,3]:
        for typ in ['test', 'dev']:
            qa_file_in = f'raw/MetaQA/{hop}-hop/vanilla/qa_{typ}.txt'
            qa_file_out = f'MetaQA_CG/{typ}/{hop}-hop_qa.jsonl'
            
                
            qa_dict_ = {}
            with open(qa_file_in,'r') as f_in:
                id_ = -1
                for qa_file_in in tqdm(f_in):#每一行
                    id_ += 1
                    que, ans = qa_file_in.strip().split('\t')
                    ans = ans.split('|')
                    qa_dict_[id_] = {'question':que, 'answer':ans}

            with open(qa_file_out,'w') as fp:
                json.dump(qa_dict_, fp)
        


if __name__=='__main__':
    # deal_metaQA()
    # deal_spider()
    # deal_wtq()
    # deal_tabfact()
    # deal_wqsp()
    deal_temp()