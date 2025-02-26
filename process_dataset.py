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
    """获取区块链和玄学相关的函数定义"""
    return [
        {
            "name": "analyze_bazi",
            "description": "分析八字与项目发展",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "项目名称"},
                    "launch_time": {"type": "string", "description": "项目启动时间"},
                    "analysis_type": {"type": "string", "description": "分析类型 (development/token/team)"}
                },
                "required": ["project_name", "launch_time"]
            }
        },
        {
            "name": "read_tarot",
            "description": "塔罗牌预测",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "预测问题"},
                    "spread_type": {"type": "string", "description": "牌阵类型"},
                    "focus_area": {"type": "string", "description": "关注领域 (investment/development/partnership)"}
                },
                "required": ["question"]
            }
        },
        {
            "name": "consult_iching",
            "description": "易经卦象解读",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "咨询问题"},
                    "hexagram": {"type": "string", "description": "卦象"},
                    "aspect": {"type": "string", "description": "关注方面 (market/technology/timing)"}
                },
                "required": ["question"]
            }
        },
        {
            "name": "analyze_astro",
            "description": "星盘分析",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_time": {"type": "string", "description": "分析时间"},
                    "aspect_type": {"type": "string", "description": "相位类型"},
                    "focus": {"type": "string", "description": "关注重点 (market_trend/token_price/community)"}
                },
                "required": ["chart_time"]
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
        }
    ]

def combine_datasets():
    """组合多个数据集"""
    try:
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logging.error("未找到 HF_TOKEN 环境变量，请确保已设置")
            return None
        
        datasets = []
        
        # 只加载基础对话数据集
        try:
            base_dataset = load_dataset("tatsu-lab/alpaca", split="train", token=hf_token)
            datasets.append(base_dataset)
            logging.info("已加载基础对话数据集")
        except Exception as e:
            logging.error(f"加载基础对话数据集失败: {str(e)}")
            return None
        
        if not datasets:
            logging.error("所有数据集加载失败")
            return None
        
        return datasets
    except Exception as e:
        logging.error(f"加载数据集时出错: {str(e)}")
        return None

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
        
        with open('data/agent_training_data.jsonl', 'w', encoding='utf-8') as f:
            for dataset in datasets:
                for item in dataset:
                    try:
                        # 根据不同数据集格式获取指令和响应
                        instruction = None
                        response = None
                        
                        # 处理基础对话数据集
                        if "instruction" in item and "output" in item:
                            instruction = item["instruction"]
                            response = item["output"]
                        # 处理新闻数据集
                        elif "text" in item:
                            instruction = f"分析这条加密货币新闻的市场影响：{item['text'][:200]}"
                            response = f"根据新闻内容，结合玄学分析，我认为这个消息对市场的影响是..."
                        # 处理基本面数据集
                        elif "news" in item:
                            instruction = f"请分析这个加密货币项目的基本面：{item['news'][:200]}"
                            response = f"从八字和星盘分析来看，这个项目的发展趋势..."
                        
                        if not instruction or not response:
                            error_count += 1
                            continue
                            
                        # 创建对话格式
                        conversation = {
                            "conversations": [
                                {"role": "user", "content": instruction},
                                {"role": "assistant", "content": "我将为您提供玄学与市场分析的综合解读。"},
                                {"role": "function_call", "content": json.dumps({
                                    "name": random.choice([f["name"] for f in functions]),
                                    "arguments": {
                                        "project_name": "Example Project",
                                        "launch_time": "2024-03-15 14:30:00",
                                        "question": instruction,
                                        "chart_time": "2024-03-15 14:30:00",
                                        "focus": "market_trend"
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
                            "system": "你是一个专业的区块链AI Agent，擅长结合玄学（八字、易经、塔罗牌、星座）和市场分析来提供独特的见解。你会谨慎评估每个预测和建议，确保分析的全面性和可靠性。"
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