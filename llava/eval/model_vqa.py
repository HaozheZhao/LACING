import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.use_fast_v == True:
        model.config.use_fast_v = True
        model.config.fast_v_inplace = args.fast_v_inplace
        model.config.fast_v_sys_length = args.fast_v_sys_length
        model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.config.fast_v_agg_layer = args.fast_v_agg_layer
    else:
        model.config.use_fast_v = False

    model.model.reset_fastv()

    model.eval()
    
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
            if args.use_uncond_embedding:
                image_tensor_cd = torch.zeros_like(image_tensor) + model.uncond_embedding.to(image_tensor.device)
        else:
            image_tensor_cd = None    
            
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                cfg_scale = args.cfg_scale,
                cfg_interval = args.cfg_interval,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_uncond_embedding=args.use_uncond_embedding,
                use_cache=args.use_cache,
                output_attentions=args.output_attentions,
                                use_dd=args.use_dd,
                use_dd_unk=args.use_dd_unk,
                # output_scores=args.output_scores,
                # return_dict_in_generate=args.return_dict_in_generate
                )
            output_ids[output_ids == -200] = 0  

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use_cache", action="store_true")
    
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    
    parser.add_argument("--cfg_scale",  type=float, default=1.0)
    parser.add_argument("--cfg_interval",  type=float, default=-1)
    parser.add_argument("--use_uncond_embedding", action='store_true', default=False)
            # vdd 
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--use_dd", action='store_true', default=False)
    #   Fast V
    parser.add_argument('--use-fast-v', default=False, action='store_true', help='whether to use fast-v')
    parser.add_argument('--fast-v-inplace', default=False, action='store_true', help='whether to use fast-v inplace version to check real latency, no supported kv cache yet')
    parser.add_argument('--fast-v-sys-length', type=int, required=False, help='the length of system prompt')
    parser.add_argument('--fast-v-image-token-length', type=int, required=False, help='the length of image token')
    parser.add_argument('--fast-v-attention-rank', type=int, required=False, help='the rank of attention matrix')
    parser.add_argument('--fast-v-agg-layer', type=int, required=False, help='the layer of attention matrix')
    parser.add_argument('--output_attentions', default=False, action='store_true')
    parser.add_argument('--output_scores', default=False, action='store_true')
    parser.add_argument('--return_dict_in_generate', default=False, action='store_true')

    args = parser.parse_args()
    if args.use_cd:
        print("start the generation using the VCD ")
        from .vcd_utils.vcd_add_noise import add_diffusion_noise
        from .vcd_utils.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()
    elif args.use_dd:
        print("start the generation using the vdd ")
        from .vcd_utils.vcd_add_noise import add_diffusion_noise
        from .vcd_utils.vdd_sample import evolve_vdd_sampling
        evolve_vdd_sampling()
    if args.cfg_scale >1:
        print("start the generation using the uncond_embedding ")
        from .uncond_utils.cfg_sample import evolve_cfg_sampling
        evolve_cfg_sampling()

    eval_model(args)