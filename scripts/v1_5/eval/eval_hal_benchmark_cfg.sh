#!/bin/bash  

# Check if the correct number of arguments is provided  
if [ "$#" -ne 2 ]; then  
    echo "Usage: $0 <model_path> <experiment_prefix>"  
    exit 1  
fi  

MODEL="$1"  
EXPERIMENT="$2"  

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  

# Declare the temperatures and evaluation types  
declare -a temperatures=("0.0" "1.0")  
declare -a eval_types=("greedy" "do_sample")  
for cfg_scale in $(seq 1.0 0.25 3.0); do  
    for i in "${!temperatures[@]}"; do  
        temperature="${temperatures[$i]}"  
        EVAL_TYPE="${eval_types[$i]}"  
        
        EXPERIMENT_NAME="${EXPERIMENT}_${EVAL_TYPE}_${cfg_scale}"  
        
        echo "Running ${cfg_scale} with temperature=${temperature} and EVAL_TYPE=${EVAL_TYPE} in objhal"  
        python -m llava.eval.model_vqa_loader_mmhal \
            --model-path "$MODEL" \
            --question-file ./playground/data/eval/objhal/obj_halbench_300_with_image.jsonl \
            --answers-file "./playground/data/eval/objhal/answers/${EXPERIMENT_NAME}.jsonl" \
            --temperature "$temperature" \
            --conv-mode vicuna_v1 \
            --use_cache \
            --cfg_scale $cfg_scale

        review_file_name=hall_obj_halbench_answer_-1.json
        coco_annotation_path=$3

        python llava/eval/eval_gpt_obj_halbench.py \
            --coco_path ./playground/data/eval/objhal/coco2014/annotations \
            --cap_folder ./playground/data/eval/objhal/answers \
            --cap_type ${EXPERIMENT_NAME}.jsonl \
            --org_folder ./playground/data/eval/objhal/obj_halbench_300_with_image.jsonl \
            --use_gpt \

        python llava/eval/summarize_gpt_obj_halbench_review.py ./playground/data/eval/objhal/answers > ./playground/data/eval/objhal/answers/${EXPERIMENT_NAME}_obj_halbench_scores.txt

        # Print Log
        echo Scores are:
        cat ./playground/data/eval/objhal/answers/${EXPERIMENT_NAME}_obj_halbench_scores.txt
        echo done


        echo "Running ${cfg_scale} with temperature=${temperature} and EVAL_TYPE=${EVAL_TYPE} in mmhal"  
        python -m llava.eval.model_vqa_loader_mmhal \
            --model-path "$MODEL" \
            --question-file ./playground/data/eval/mmhal/mmhal-bench_with_image.jsonl \
            --answers-file "./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl" \
            --temperature "$temperature" \
            --conv-mode vicuna_v1 \
            --use_cache \
            --cfg_scale $cfg_scale


        python llava/eval/change_mmhal_predict_template.py \
            --response-template ./playground/data/eval/mmhal/mmhal-bench_answer_template.json \
            --answers-file ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl \
            --save-file ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl.template.json

        python llava/eval/eval_gpt_mmhal.py \
            --response ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl.template.json \
            --evaluation ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl.mmhal_test_eval.json \
        >> ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl.eval_log.txt

        # Merge gpt4 evaluation to the original model outputs, can be ignore
        python llava/eval/merge_mmhal_review_with_predict.py \
            --review_path ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl.mmhal_test_eval.json \
            --predict_path ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl \
            --save_path ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}.jsonl.mmhal_test_all_infos.json

        python llava/eval/summarize_gpt_mmhal_review.py ./playground/data/eval/mmhal/answers > ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}_mmhal_scores.txt

        # Print Log
        echo Scores are:
        cat ./playground/data/eval/mmhal/answers/${EXPERIMENT_NAME}_mmhal_scores.txt
        echo done

    done
done
