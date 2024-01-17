OUTPUTPATH=output/CronQuestion/4
mkdir -p $OUTPUTPATH

python CGdata_for_cronquestion.py \
--key api_key.txt --num_process 5 \
--folder_path dataset/CronQuestion_CG/kg.txt \
--data_path dataset/CronQuestion_CG/qa_test.jsonl \
--prompt_path structllm/prompt_/CronQuestion.json \
--model gpt-4-1106-preview \
--output_detail_path $OUTPUTPATH/output_detail \
--output_result_path $OUTPUTPATH/output_result \
--SC_Num 5 \
--debug 0

cat $OUTPUTPATH/output_detail* > $OUTPUTPATH/all_detail.txt
cat $OUTPUTPATH/output_result* > $OUTPUTPATH/all_result.txt

python evaluate/evaluate_for_CronQuestion.py \
--ori_path $OUTPUTPATH/all_result.txt\
--error_cases_output $OUTPUTPATH/error_cases.txt \
--write_flag True