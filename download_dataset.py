from datasets import load_dataset
import json
import random
import os
import logging
from typing import Dict, List
import time

# 配置日志
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
        }
    ]

def convert_to_agent_format(example: Dict) -> Dict:
    """将原始数据转换为agent格式"""
    try:
        conversation = []
        functions = get_blockchain_functions()
        
        # 添加用户输入
        conversation.append({
            "role": "user",
            "content": example["instruction"]
        })
        
        # 添加助手回应
        conversation.append({
            "role": "assistant",
            "content": "我将帮您完成这个任务。让我分析一下需求。"
        })
        
        # 添加function calling
        selected_function = random.choice(functions)
        function_call = {
            "name": selected_function["name"],
            "arguments": {
                "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e" if "address" in selected_function["parameters"]["required"] else None,
                "protocols": ["Aave", "Uniswap"] if "protocols" in selected_function["parameters"]["required"] else None,
                "from_token": "ETH" if "from_token" in selected_function["parameters"]["required"] else None,
                "to_token": "USDC" if "to_token" in selected_function["parameters"]["required"] else None,
                "amount": "1.0" if "amount" in selected_function["parameters"]["required"] else None
            }
        }
        
        # 移除None值的参数
        function_call["arguments"] = {k: v for k, v in function_call["arguments"].items() if v is not None}
        
        conversation.append({
            "role": "function_call",
            "content": json.dumps(function_call)
        })
        
        # 添加observation
        observation_result = {
            "status": "success",
            "data": example["output"],
            "timestamp": int(time.time())
        }
        conversation.append({
            "role": "observation",
            "content": json.dumps(observation_result)
        })
        
        # 添加最终回应
        conversation.append({
            "role": "assistant",
            "content": example["output"]
        })
        
        return {
            "conversations": conversation,
            "tools": json.dumps(functions),
            "system": "你是一个专业的区块链AI Agent，擅长分析和执行各种区块链操作。你会仔细评估每个操作的风险，并确保用户资产的安全。"
        }
    except Exception as e:
        logging.error(f"转换数据时出错: {str(e)}")
        return None

def download_and_process_dataset():
    try:
        # 检查环境变量
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logging.error("未找到HF_TOKEN环境变量")
            return
        
        # 加载数据集
        logging.info("正在下载数据集...")
        dataset = load_dataset("tatsu-lab/alpaca", use_auth_token=hf_token)
        
        logging.info("正在处理数据...")
        output_file = 'data/agent_training_data.jsonl'
        processed_count = 0
        error_count = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 处理前5000条数据
            for item in list(dataset['train'])[:5000]:
                try:
                    conversation = convert_to_agent_format(item)
                    if conversation:
                        f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logging.error(f"处理数据时出错: {str(e)}")
                    error_count += 1
                
                if processed_count % 100 == 0:
                    logging.info(f"已处理 {processed_count} 条数据...")
        
        logging.info(f"数据处理完成！成功: {processed_count}, 失败: {error_count}")
        logging.info(f"数据已保存到: {output_file}")
        
    except Exception as e:
        logging.error(f"处理数据集时出错: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 确保data目录存在
        os.makedirs('data', exist_ok=True)
        
        logging.info("开始下载和处理数据集...")
        download_and_process_dataset()
        logging.info("全部处理完成！")
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        raise 