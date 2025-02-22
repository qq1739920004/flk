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
        
        try:
            # 加载塔罗牌数据集
            tarot_dataset = load_dataset("astro-gpt/tarot-readings", split="train", token=hf_token)
            datasets.append(tarot_dataset)
            logging.info("已加载塔罗牌数据集")
        except Exception as e:
            logging.error(f"加载塔罗牌数据集失败: {str(e)}")
            
        try:
            # 加载八字数据集
            bazi_dataset = load_dataset("astro-gpt/bazi-analysis", split="train", token=hf_token)
            datasets.append(bazi_dataset)
            logging.info("已加载八字数据集")
        except Exception as e:
            logging.error(f"加载八字数据集失败: {str(e)}")
            
        try:
            # 加载易经数据集
            iching_dataset = load_dataset("astro-gpt/iching-readings", split="train", token=hf_token)
            datasets.append(iching_dataset)
            logging.info("已加载易经数据集")
        except Exception as e:
            logging.error(f"加载易经数据集失败: {str(e)}")
            
        try:
            # 加载加密货币新闻数据集
            news_dataset = load_dataset("SahandNZ/cryptonews-articles-with-price-momentum-labels", split="train", token=hf_token)
            datasets.append(news_dataset)
            logging.info("已加载加密货币新闻数据集")
        except Exception as e:
            logging.error(f"加载加密货币新闻数据集失败: {str(e)}")
            
        try:
            # 加载加密货币基本面数据集
            fundamental_dataset = load_dataset("arad1367/Crypto_Fundamental_News", split="train", token=hf_token)
            datasets.append(fundamental_dataset)
            logging.info("已加载加密货币基本面数据集")
        except Exception as e:
            logging.error(f"加载加密货币基本面数据集失败: {str(e)}")
            
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
                            item.get("reading") or  # 适配占卜类数据集
                            item.get("analysis") or # 适配分析类数据集
                            item.get("interpretation")  # 适配易经解读
                        )
                        
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