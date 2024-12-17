#!/bin/bash  

# Check if the correct number of arguments is provided  
if [ "$#" -ne 2 ]; then  
    echo "Usage: $0 <model_path> <experiment_prefix>"  
    exit 1  
fi  

MODEL="$1"  
EXPERIMENT="$2"  

declare -a temperatures=("0.0" "1.0")  
declare -a eval_types=("greedy" "do_sample")  
for cfg_scale in $(seq 1.0 0.25 2.5); do  
    for i in "${!temperatures[@]}"; do  
        temperature="${temperatures[$i]}"  
        EVAL_TYPE="${eval_types[$i]}"  
        
        EXPERIMENT_NAME="${EXPERIMENT}_${EVAL_TYPE}_${cfg_scale}"  
        #################################################################################################################
        echo "Running with temperature=${temperature} and EVAL_TYPE=${EVAL_TYPE} in LLavaBench"  
        python -m llava.eval.model_vqa \
            --model-path "$MODEL" \
            --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
            --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
            --answers-file "./playground/data/eval/llava-bench-in-the-wild/answers/${EXPERIMENT_NAME}.jsonl" \
            --temperature "$temperature" \
            --conv-mode vicuna_v1 \
            --use_cache \
            --cfg_scale $cfg_scale

        #################################################################################################################

        echo "Running with temperature=${temperature} and EVAL_TYPE=${EVAL_TYPE} in MMVET"  
        python -m llava.eval.model_vqa \
            --model-path "$MODEL" \
            --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
            --image-folder ./playground/data/eval/mm-vet/images \
            --answers-file "./playground/data/eval/mm-vet/answers/${EXPERIMENT_NAME}.jsonl" \
            --temperature "$temperature" \
            --conv-mode vicuna_v1 \
            --use_cache \
            --cfg_scale $cfg_scale

        # Convert for evaluation  
        python scripts/convert_mmvet_for_eval.py \
            --src "./playground/data/eval/mm-vet/answers/${EXPERIMENT_NAME}.jsonl" \
            --dst "./playground/data/eval/mm-vet/results/${EXPERIMENT_NAME}.json"  

        #################################################################################################################

        echo "Running with temperature=${temperature} and EVAL_TYPE=${EVAL_TYPE} in TEXTVQA without OCR Context"  
        python -m llava.eval.model_vqa_loader \
            --model-path "$MODEL" \
            --question-file ./playground/data/eval/textvqa/modified_llava_textvqa_val_v051_ocr.jsonl \
            --image-folder ./playground/data/textvqa/train_images \
            --answers-file "./playground/data/eval/textvqa/answers/${EXPERIMENT_NAME}_wo_ocr.jsonl" \
            --temperature "$temperature" \
            --conv-mode vicuna_v1 \
            --use_cache \
            --cfg_scale $cfg_scale

        # Convert for evaluation  
        python -m llava.eval.eval_textvqa \
            --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
            --result-file "./playground/data/eval/textvqa/answers/${EXPERIMENT_NAME}_wo_ocr.jsonl"  

        #################################################################################################################

    #################################################################################################################

        SPLIT="mmbench_dev_20230712"

        echo "${EXPERIMENT_NAME}  Running with temperature=${temperature} and EVAL_TYPE=${EVAL_TYPE} in MMBENCH"  
        python -m llava.eval.model_vqa_mmbench \
            --model-path "$MODEL" \
            --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
            --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${EXPERIMENT_NAME}.jsonl \
            --single-pred-prompt \
            --temperature "$temperature" \
            --conv-mode vicuna_v1 \
            --cfg_scale $cfg_scale


        mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

        python scripts/convert_mmbench_for_submission.py \
            --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
            --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
            --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
            --experiment $EXPERIMENT_NAME \

    done
done
