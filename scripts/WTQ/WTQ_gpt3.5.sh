OUTPUTPATH=output/WTQ/3.5
mkdir -p $OUTPUTPATH

python CGdata_for_tableqa.py \
--key api_key.txt --num_process 5 \
--folder_path dataset/WTQ/csv \
--data_path dataset/WTQ/test.jsonl \
--prompt_path structllm/prompt_/WTQ.json \
--model gpt-3.5-turbo \
--output_detail_path $OUTPUTPATH/output_detail \
--output_result_path $OUTPUTPATH/output_result \
--SC_Num 5 \
--debug 0

cat $OUTPUTPATH/output_detail* > $OUTPUTPATH/all_detail.txt
cat $OUTPUTPATH/output_result* > $OUTPUTPATH/all_result.txt

python evaluate/evaluate_for_tableqa.py \
--ori_path $OUTPUTPATH/all_result.txt\
--error_cases_output $OUTPUTPATH/error_cases.txt \
--write_flag True