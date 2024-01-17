scripts/WQSP/wqsp3.5/WQSP_name_gpt3.5.sh
scripts/WQSP/wqsp3.5/WQSP_unname_gpt3.5.sh

OUTPUTPATH=output/WQSP/gpt3.5
mkdir -p $OUTPUTPATH

cat $OUTPUTPATH/*/all_detail.txt > $OUTPUTPATH/all_detail.txt
cat $OUTPUTPATH/*/all_result.txt > $OUTPUTPATH/all_result.txt

python evaluate/evaluate_for_wqsp.py \
--ori_path $OUTPUTPATH/all_result.txt\
--error_cases_output $OUTPUTPATH/error_cases.txt \
--write_flag True