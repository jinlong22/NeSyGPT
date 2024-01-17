scripts/WQSP/wqsp4/WQSP_name_gpt4.sh
scripts/WQSP/wqsp4/WQSP_unname_gpt4.sh

OUTPUTPATH=output/WQSP/gpt4
mkdir -p $OUTPUTPATH

cat $OUTPUTPATH/*/all_detail.txt > $OUTPUTPATH/all_detail.txt
cat $OUTPUTPATH/*/all_result.txt > $OUTPUTPATH/all_result.txt

python evaluate/evaluate_for_wqsp.py \
--ori_path $OUTPUTPATH/all_result.txt\
--error_cases_output $OUTPUTPATH/error_cases.txt \
--write_flag True