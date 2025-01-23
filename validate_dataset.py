import json
from typing import List, Dict
import os

def validate_conversation(conversation: Dict) -> List[str]:
    errors = []
    required_fields = ['conversations', 'tools', 'system']
    
    # 检查必需字段
    for field in required_fields:
        if field not in conversation:
            errors.append(f"缺少必需字段: {field}")
    
    if 'conversations' in conversation:
        # 检查对话格式
        for turn in conversation['conversations']:
            if not isinstance(turn, dict):
                errors.append("对话格式错误：每个对话轮次应该是字典格式")
                continue
            
            if 'role' not in turn or 'content' not in turn:
                errors.append("对话缺少必需的role或content字段")
            
            if turn.get('role') not in ['user', 'assistant', 'function_call', 'observation']:
                errors.append(f"无效的对话角色: {turn.get('role')}")
    
    # 验证tools格式
    if 'tools' in conversation:
        try:
            tools = json.loads(conversation['tools'])
            if not isinstance(tools, list):
                errors.append("tools字段应该是JSON数组格式")
        except json.JSONDecodeError:
            errors.append("tools字段不是有效的JSON格式")
    
    return errors

def validate_dataset(file_path: str):
    print(f"开始验证数据集: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 {file_path}")
        return
    
    total_conversations = 0
    error_conversations = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                conversation = json.loads(line)
                errors = validate_conversation(conversation)
                
                if errors:
                    error_conversations += 1
                    print(f"\n行号 {line_num} 存在以下问题：")
                    for error in errors:
                        print(f"- {error}")
                
                total_conversations += 1
                
            except json.JSONDecodeError:
                error_conversations += 1
                print(f"\n行号 {line_num} JSON解析错误")
    
    print(f"\n验证完成:")
    print(f"总对话数: {total_conversations}")
    print(f"有问题的对话数: {error_conversations}")
    print(f"成功率: {((total_conversations - error_conversations) / total_conversations * 100):.2f}%")

if __name__ == "__main__":
    validate_dataset('data/agent_training_data.jsonl') 