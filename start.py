import sys
from vllm.entrypoints.cli.main import main
import transformers.processing_utils as pu

# 1. 保存原函数
original_from_args_and_dict = pu.ProcessorMixin.from_args_and_dict

# 2. 定义安全拦截器
@classmethod
def patched_from_args_and_dict(cls, args, processor_dict, **kwargs):
    # 如果 kwargs 里有 tokenizer 或 image_processor，并且位置参数 args 里也传了，就删掉 kwargs 里的
    if 'tokenizer' in kwargs and len(args) > 0:
        kwargs.pop('tokenizer')
    if 'image_processor' in kwargs and len(args) > 1:
        kwargs.pop('image_processor')
        
    # 注意这里：直接调用 __func__ 来避开 classmethod 的自动绑定问题
    return original_from_args_and_dict.__func__(cls, args, processor_dict, **kwargs)

# 3. 替换原函数
pu.ProcessorMixin.from_args_and_dict = patched_from_args_and_dict

# 4. 配置启动参数
sys.argv = [
    "vllm", "serve",
    "/tmp/AirNav/model_weight/AirVLN-R1",
    "--dtype", "auto",
    "--served-model-name", "qwen_2_5_vl_7b",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "1",
    "--limit-mm-per-prompt", "image=5",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--max-num-seqs", "1"
]

if __name__ == "__main__":
    print("✅ 终极防冲突补丁注入成功，正在启动事件循环...")
    main()