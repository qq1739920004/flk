from datasets import load_dataset
import json
import random
import os
import logging
from typing import Dict, List
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/dataset_processing.log'),
        logging.StreamHandler()
    ]
)

def get_blockchain_functions() -> List[Dict]:
    """获取区块链相关的函数定义"""
    return [
        {
            "name": "get_eth_balance",
            "description": "获取ETH余额",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "钱包地址"},
                    "block_number": {"type": "string", "description": "区块高度，可选"}
                },
                "required": ["address"]
            }
        },
        {
            "name": "analyze_defi_risk",
            "description": "分析DeFi投资组合风险",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "钱包地址"},
                    "protocols": {"type": "array", "items": {"type": "string"}, "description": "要分析的协议列表"},
                    "time_range": {"type": "string", "description": "分析时间范围"}
                },
                "required": ["address", "protocols"]
            }
        },
        {
            "name": "execute_swap",
            "description": "在DEX上执行代币兑换",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_token": {"type": "string", "description": "源代币"},
                    "to_token": {"type": "string", "description": "目标代币"},
                    "amount": {"type": "string", "description": "兑换数量"},
                    "slippage": {"type": "string", "description": "滑点限制"}
                },
                "required": ["from_token", "to_token", "amount"]
            }
        },
        {
            "name": "monitor_contract",
            "description": "监控智能合约活动",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_address": {"type": "string", "description": "合约地址"},
                    "event_types": {"type": "array", "items": {"type": "string"}, "description": "要监控的事件类型"},
                    "threshold": {"type": "string", "description": "警报阈值"}
                },
                "required": ["contract_address"]
            }
        },
        {
            "name": "deploy_contract",
            "description": "部署智能合约",
            "parameters": {
                "type": "object",
                "properties": {
                    "contract_type": {"type": "string", "description": "合约类型 (ERC20/ERC721/Custom)"},
                    "constructor_args": {"type": "object", "description": "构造函数参数"},
                    "network": {"type": "string", "description": "目标网络"}
                },
                "required": ["contract_type", "constructor_args"]
            }
        },
        {
            "name": "analyze_transaction",
            "description": "分析交易详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "tx_hash": {"type": "string", "description": "交易哈希"},
                    "analysis_type": {"type": "string", "description": "分析类型 (gas/trace/impact)"}
                },
                "required": ["tx_hash"]
            }
        }
    ]

def combine_datasets():
    """组合多个数据集"""
    try:
        datasets = []
        
        # 使用环境变量中的 token
        token = os.getenv('HF_TOKEN')
        if not token:
            logging.error("请提供 HF_TOKEN 环境变量")
            return None
            
        # 加载数据集时使用 token
        auth_token = {"use_auth_token": token}
        
        # 加载 Human-Like-DPO 数据集
        dpo_dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train", **auth_token)
        datasets.append(dpo_dataset)
        logging.info("已加载 Human-Like-DPO 数据集")
        
        # 加载 awesome-chatgpt-prompts 数据集
        prompts_dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train", **auth_token)
        datasets.append(prompts_dataset)
        logging.info("已加载 awesome-chatgpt-prompts 数据集")
        
        # 加载 agent-instruction 数据集
        agent_dataset = load_dataset("jinaai/agent-instruction-dataset", split="train", **auth_token)
        datasets.append(agent_dataset)
        logging.info("已加载 agent-instruction 数据集")
        
        # 加载 multimodal_textbook 数据集
        textbook_dataset = load_dataset("DAMO-NLP-SG/multimodal_textbook", split="train", **auth_token)
        datasets.append(textbook_dataset)
        logging.info("已加载 multimodal_textbook 数据集")
        
        # 加载原有的数据集
        code_dataset = load_dataset("jinaai/code_exercises", split="train", **auth_token)
        datasets.append(code_dataset)
        logging.info("已加载 code_exercises 数据集")
        
        reader_dataset = load_dataset("jinaai/ReaderLM-v2", split="train", **auth_token)
        datasets.append(reader_dataset)
        logging.info("已加载 ReaderLM-v2 数据集")
        
        return datasets
    except Exception as e:
        logging.error(f"加载数据集时出错: {str(e)}")
        return None

def get_example_values():
    """获取示例值，避免硬编码"""
    return {
        "example_address": "0x0000000000000000000000000000000000000000",
        "example_contract": "0x0000000000000000000000000000000000000000",
        "example_tx_hash": "0x0000000000000000000000000000000000000000000000000000000000000000"
    }

def process_dataset():
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logging.error("未找到HF_TOKEN环境变量，请在运行时提供")
            return False
            
        logging.info("开始加载数据集...")
        datasets = combine_datasets()
        if not datasets:
            logging.error("无法加载数据集")
            return False
            
        functions = get_blockchain_functions()
        processed_count = 0
        error_count = 0
        example_values = get_example_values()
        
        with open('data/agent_training_data.jsonl', 'w', encoding='utf-8') as f:
            for dataset in datasets:
                for item in dataset:
                    try:
                        instruction = (
                            item.get("instruction") or 
                            item.get("input") or 
                            item.get("question") or
                            item.get("prompt") or
                            item.get("text")
                        )
                        response = (
                            item.get("output") or 
                            item.get("response") or 
                            item.get("answer") or
                            item.get("completion")
                        )
                        
                        if not instruction or not response:
                            error_count += 1
                            continue
                            
                        conversation = {
                            "conversations": [
                                {"role": "user", "content": instruction},
                                {"role": "assistant", "content": "我将帮您处理这个请求。"},
                                {"role": "function_call", "content": json.dumps({
                                    "name": random.choice([f["name"] for f in functions]),
                                    "arguments": {
                                        "address": example_values["example_address"],
                                        "contract_address": example_values["example_contract"],
                                        "tx_hash": example_values["example_tx_hash"]
                                    }
                                })},
                                {"role": "observation", "content": json.dumps({
                                    "status": "success",
                                    "data": response,
                                    "timestamp": int(time.time())
                                })},
                                {"role": "assistant", "content": response}
                            ],
                            "tools": json.dumps(functions),
                            "system": "你是一个专业的区块链AI Agent，擅长分析和执行各种区块链操作。你会仔细评估每个操作的风险，并确保用户资产的安全。"
                        }
                        
                        f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            logging.info(f"已处理 {processed_count} 条数据...")
                            
                    except Exception as e:
                        error_count += 1
                        logging.error(f"处理数据时出错: {str(e)}")
                        continue
        
        logging.info(f"数据处理完成！成功处理 {processed_count} 条数据，失败 {error_count} 条")
        return True
        
    except Exception as e:
        logging.error(f"处理数据集时出错: {str(e)}")
        return False

if __name__ == "__main__":
    process_dataset() 