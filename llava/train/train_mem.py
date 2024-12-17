from llava.train.train import train
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()
if __name__ == "__main__":
    train(attn_implementation="eager")
