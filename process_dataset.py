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
        # 检查环境变量
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logging.error("未找到 HF_TOKEN 环境变量，请确保已设置")
            return None
            
        datasets = []
        
        try:
            # 只加载 glaive function calling 数据集
            glaive_dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train", token=hf_token)
            datasets.append(glaive_dataset)
            logging.info("已加载 glaive-function-calling 数据集")
        except Exception as e:
            logging.error(f"加载 glaive-function-calling 数据集失败: {str(e)}")
        
        # 如果数据集加载失败，使用本地示例数据
        if not datasets:
            logging.warning("glaive数据集加载失败，使用本地示例数据")
            example_data = [
                {
                    "instruction": "分析这个地址的DeFi活动：0x742d35Cc6634C0532925a3b844Bc454e4438f44e，我想知道它在过去30天内的收益率和风险指标",
                    "output": "根据分析，这个地址在DeFi上的活动情况如下：\n\n1. Aave协议活动：\n- 存款：10,000 USDC (3.5% APY) 和 5 ETH (2.1% APY)\n- 借款：5,000 DAI (4.2% APY)\n\n2. Uniswap流动性提供：\n- ETH/USDC池：价值20,000美元\n- 已赚取手续费：150美元\n\n风险指标：\n- 健康因子：1.8（安全范围内，>1.5为良好）\n- 清算阈值：80%\n- 波动性风险：中等\n- 无常损失风险：高\n\n总体年化收益率：5.2%"
                },
                {
                    "instruction": "帮我分析这笔交易的gas使用情况：0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                    "output": "这笔交易的gas分析如下：\n\n1. Gas使用明细：\n- 基础gas费用：21,000\n- 合约调用gas：45,000\n- 存储操作gas：15,000\n总gas使用：81,000\n\n2. Gas价格：\n- 基础费用：25 Gwei\n- 优先费用：2 Gwei\n总gas价格：27 Gwei\n\n3. 总费用：0.002187 ETH (约 $4.50)\n\n4. 优化建议：\n- 可以通过批量处理减少gas使用\n- 建议在网络不拥堵时执行交易"
                },
                {
                    "instruction": "我想部署一个ERC20代币合约，代币名称是'MyToken'，符号是'MTK'，总供应量是1000000",
                    "output": "我将帮您部署ERC20代币合约。以下是部署详情：\n\n1. 合约参数：\n- 名称：MyToken\n- 符号：MTK\n- 小数位：18\n- 总供应量：1,000,000 MTK\n\n2. 部署网络：Ethereum主网\n\n3. 预估gas费用：\n- 合约部署约需要 800,000 gas\n- 当前gas价格：30 Gwei\n- 总费用约：0.024 ETH\n\n4. 安全提示：\n- 合约已包含权限控制\n- 实现了暂停功能\n- 添加了事件日志"
                }
            ]
            from datasets import Dataset
            datasets = [Dataset.from_list(example_data)]
            
        return datasets
    except Exception as e:
        logging.error(f"加载数据集时出错: {str(e)}")
        return None

def process_dataset():
    try:
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)  # 确保日志目录存在
        
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
                            item.get("prompt") or  # 适配 awesome-chatgpt-prompts
                            item.get("text")  # 适配其他可能的格式
                        )
                        response = (
                            item.get("output") or 
                            item.get("response") or 
                            item.get("answer") or
                            item.get("completion")  # 适配 awesome-chatgpt-prompts
                        )
                        
                        if not instruction or not response:
                            error_count += 1
                            continue
                            
                        # 创建对话格式
                        conversation = {
                            "conversations": [
                                {"role": "user", "content": instruction},
                                {"role": "assistant", "content": "我将帮您处理这个请求。"},
                                {"role": "function_call", "content": json.dumps({
                                    "name": random.choice([f["name"] for f in functions]),
                                    "arguments": {
                                        "address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e",
                                        "contract_address": "0x1234567890abcdef1234567890abcdef12345678",
                                        "tx_hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
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