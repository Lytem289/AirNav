import asyncio
import os

# 彻底禁用 uvloop 策略（如果还没卸载干净）
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from verl.trainer.benchmark_trainer import main

if __name__ == "__main__":
    # 在主进程创建并设置 loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 强制让 Ray 节点也尝试继承某些环境
    os.environ['RAY_worker_register_timeout_seconds'] = '600'
    asyncio.get_event_loop_policy().get_event_loop()
    
    main()