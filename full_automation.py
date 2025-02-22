import json
import os
import sys

import requests
import yaml
from loguru import logger
from huggingface_hub import HfApi

from demo import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

HF_USERNAME = os.environ["HF_USERNAME"]

if __name__ == "__main__":
    task_id = 5
    
    # load training args
    current_folder = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_folder}/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)
    
    # 如果设置了MODEL_ID环境变量，就只使用指定的模型
    if "MODEL_ID" in os.environ:
        model_id = os.environ["MODEL_ID"]
        if model_id in all_training_args:
            all_training_args = {model_id: all_training_args[model_id]}
        else:
            logger.error(f"Model {model_id} not found in training_args.yaml")
            sys.exit(1)

    try:
        # 获取任务信息
        task = get_task(task_id)
        logger.info(f"Retrieved task: {task}")
        
        if 'data' not in task:
            logger.error(f"Task does not contain 'data' field. Task content: {task}")
            sys.exit(1)
        
        data_url = task["data"]["training_set_url"]
        context_length = task["data"]["context_length"]
        max_params = task["data"]["max_params"]

        # filter out the model within the max_params
        model2size = {k: v for k, v in model2size.items() if v <= max_params}
        all_training_args = {k: v for k, v in all_training_args.items() if k in model2size}
        logger.info(f"Models within the max_params: {all_training_args.keys()}")
        
        # 下载任务数据
        response = requests.get(data_url, stream=True)
        with open("data/demo_data.jsonl", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 处理额外的训练数据
        logger.info("开始处理额外的训练数据...")
        from process_dataset import process_dataset
        if not process_dataset():
            logger.error("处理额外训练数据失败")
            sys.exit(1)
        
        # 合并数据集
        logger.info("合并数据集...")
        merged_data = []
        
        # 读取任务数据
        with open("data/demo_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                merged_data.append(line.strip())
                
        # 读取处理后的数据
        with open("data/agent_training_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                merged_data.append(line.strip())
        
        # 保存合并后的数据
        with open("data/demo_data.jsonl", "w", encoding="utf-8") as f:
            for line in merged_data:
                f.write(line + "\n")
        
        logger.info(f"数据集合并完成，共 {len(merged_data)} 条数据")

        # train all feasible models and merge
        for model_id in all_training_args.keys():
            logger.info(f"Start to train the model {model_id}...")
            # if OOM, proceed to the next model
            try:
                # 确保只传入需要的参数
                training_args = LoraTrainingArguments(**all_training_args[model_id])
                train_lora(
                    model_id=model_id,
                    context_length=context_length,
                    training_args=training_args,
                )
            except RuntimeError as e:
                logger.error(f"Error: {e}")
                logger.info("Proceed to the next model...")
                continue

            # generate a random repo id based on timestamp
            gpu_type = get_gpu_type()

            try:
                logger.info("Start to push the lora weight to the hub...")
                api = HfApi(token=os.environ["HF_TOKEN"])
                repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
                # check whether the repo exists
                try:
                    api.create_repo(
                        repo_name,
                        exist_ok=False,
                        repo_type="model",
                    )
                except Exception:
                    logger.info(
                        f"Repo {repo_name} already exists. Will commit the new version."
                    )

                commit_message = api.upload_folder(
                    folder_path="outputs",
                    repo_id=repo_name,
                    repo_type="model",
                )
                # get commit hash
                commit_hash = commit_message.oid
                logger.info(f"Commit hash: {commit_hash}")
                logger.info(f"Repo name: {repo_name}")
                # submit
                submit_task(
                    task_id, repo_name, model2base_model[model_id], gpu_type, commit_hash
                )
                logger.info("Task submitted successfully")
            except Exception as e:
                logger.error(f"Error: {e}")
                logger.info("Proceed to the next model...")
            finally:
                # cleanup merged_model and output
                os.system("rm -rf merged_model")
                os.system("rm -rf outputs")
                continue

    except KeyError as e:
        logger.error(f"Failed to access required field: {e}")
        logger.error(f"Task structure: {task}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Full task response: {task}")
        sys.exit(1)
