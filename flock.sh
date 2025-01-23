#!/bin/bash
# Miniconda安装路径
MINICONDA_PATH="$HOME/miniconda"
CONDA_EXECUTABLE="$MINICONDA_PATH/bin/conda"

# 检查是否以root用户运行脚本
if [ "$(id -u)" != "0" ]; then
    echo "此脚本需要以root用户权限运行。"
    echo "请尝试使用 'sudo -i' 命令切换到root用户，然后再次运行此脚本。"
    exit 1
fi

# 确保 conda 被正确初始化
ensure_conda_initialized() {
    if [ -f "$HOME/.bashrc" ]; then
        source "$HOME/.bashrc"
    fi
    if [ -f "$CONDA_EXECUTABLE" ]; then
        eval "$("$CONDA_EXECUTABLE" shell.bash hook)"
    fi
}

# 检查并安装 Conda
function install_conda() {
    if [ -f "$CONDA_EXECUTABLE" ]; then
        echo "Conda 已安装在 $MINICONDA_PATH"
        ensure_conda_initialized
    else
        echo "Conda 未安装，正在安装..."
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        bash miniconda.sh -b -p $MINICONDA_PATH
        rm miniconda.sh
        
        # 初始化 conda
        "$CONDA_EXECUTABLE" init
        ensure_conda_initialized
        
        echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
        source ~/.bashrc
    fi
    
    # 验证 conda 是否可用
    if command -v conda &> /dev/null; then
        echo "Conda 安装成功，版本: $(conda --version)"
    else
        echo "Conda 安装可能成功，但无法在当前会话中使用。"
        echo "请在脚本执行完成后，重新登录或运行 'source ~/.bashrc' 来激活 Conda。"
    fi
}

# 检查并安装 Node.js 和 npm
function install_nodejs_and_npm() {
    if command -v node > /dev/null 2>&1; then
        echo "Node.js 已安装，版本: $(node -v)"
    else
        echo "Node.js 未安装，正在安装..."
        curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
        sudo apt-get install -y nodejs git
    fi
    if command -v npm > /dev/null 2>&1; then
        echo "npm 已安装，版本: $(npm -v)"
    else
        echo "npm 未安装，正在安装..."
        sudo apt-get install -y npm
    fi
}

# 检查并安装 PM2
function install_pm2() {
    if command -v pm2 > /dev/null 2>&1; then
        echo "PM2 已安装，版本: $(pm2 -v)"
    else
        echo "PM2 未安装，正在安装..."
        npm install pm2@latest -g
    fi
}

function install_node() {
    apt update && apt upgrade -y
    apt install curl sudo git python3-venv iptables build-essential wget jq make gcc nano npm -y
    install_conda
    ensure_conda_initialized
    install_nodejs_and_npm
    install_pm2
    read -p "输入Hugging face API: " HF_TOKEN
    read -p "输入Flock API: " FLOCK_API_KEY
    read -p "输入任务ID: " TASK_ID
    # 克隆仓库
    git clone https://github.com/FLock-io/llm-loss-validator.git
    # 进入项目目录
    cd llm-loss-validator
    # 创建并激活conda环境
    conda create -n llm-loss-validator python==3.10 -y
    source "$MINICONDA_PATH/bin/activate" llm-loss-validator
    # 安装依赖
    pip install -r requirements.txt
    # 获取当前目录的绝对路径
    SCRIPT_DIR="$(pwd)"
    # 创建启动脚本
    cat << EOF > run_validator.sh
#!/bin/bash
source "$MINICONDA_PATH/bin/activate" llm-loss-validator
cd $SCRIPT_DIR/src
CUDA_VISIBLE_DEVICES=0 \
TIME_SLEEP=180 \
bash start.sh \
--hf_token "$HF_TOKEN" \
--flock_api_key "$FLOCK_API_KEY" \
--task_id "$TASK_ID" \
--validation_args_file validation_config.json.example \
--auto_clean_cache True
EOF
    chmod +x run_validator.sh
    pm2 start run_validator.sh --name "llm-loss-validator" -- start && pm2 save && pm2 startup
    echo "验证者节点已经启动."
}

function check_node() {
    pm2 logs llm-loss-validator
}

function uninstall_node() {
    pm2 delete llm-loss-validator && rm -rf llm-loss-validator
}

function install_train_node() {
    install_conda
    ensure_conda_initialized
    install_nodejs_and_npm
    install_pm2
    
    # 安装必要的工具
    apt update && apt upgrade -y
    apt install curl sudo python3-venv iptables build-essential wget jq make gcc nano git -y
    
    # 克隆 QuickStart 仓库
    git clone https://github.com/qq1739920004/flk.git
    cd flk
    
    # 创建并激活 conda 环境
    conda create -n training-node python==3.10 -y
    source "$MINICONDA_PATH/bin/activate" training-node
    
    # 安装依赖
    pip install -r requirements.txt
    pip install datasets  # 添加数据集处理所需的库
    
    # 获取必要信息
    read -p "输入任务ID (TASK_ID): " TASK_ID
    read -p "输入Flock API Key: " FLOCK_API_KEY
    read -p "输入Hugging Face Token: " HF_TOKEN
    read -p "输入Hugging Face 用户名: " HF_USERNAME
    
    # 创建数据处理脚本
    echo "创建数据处理脚本..."
    cat > process_dataset.py << 'EOL'
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

def process_dataset():
    try:
        os.makedirs('data', exist_ok=True)
        hf_token = os.getenv('HF_TOKEN')
        if not hf_token:
            logging.error("未找到HF_TOKEN环境变量")
            return False
        
        logging.info("开始下载和处理数据集...")
        dataset = load_dataset("tatsu-lab/alpaca", use_auth_token=hf_token)
        functions = get_blockchain_functions()
        
        with open('data/agent_training_data.jsonl', 'w', encoding='utf-8') as f:
            for item in list(dataset['train'])[:5000]:
                conversation = {
                    "conversations": [
                        {"role": "user", "content": item["instruction"]},
                        {"role": "assistant", "content": "我将帮您完成这个任务。"},
                        {"role": "function_call", "content": json.dumps({
                            "name": random.choice([f["name"] for f in functions]),
                            "arguments": {"address": "0x742d35Cc6634C0532925a3b844Bc454e4438f44e"}
                        })},
                        {"role": "observation", "content": json.dumps({"result": item["output"], "status": "success"})},
                        {"role": "assistant", "content": item["output"]}
                    ],
                    "tools": json.dumps(functions),
                    "system": "你是一个专业的区块链AI Agent，擅长分析和执行各种区块链操作。"
                }
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        logging.info("数据处理完成！")
        return True
    except Exception as e:
        logging.error(f"处理数据集时出错: {str(e)}")
        return False

if __name__ == "__main__":
    process_dataset()
EOL
    
    # 创建运行脚本
    cat << EOF > run_training_node.sh
#!/bin/bash
source "$MINICONDA_PATH/bin/activate" training-node

# 首先处理数据集
echo "开始处理训练数据集..."
python process_dataset.py

# 如果数据处理成功，启动训练节点
if [ $? -eq 0 ]; then
    echo "数据处理完成，启动训练节点..."
    TASK_ID=$TASK_ID FLOCK_API_KEY="$FLOCK_API_KEY" HF_TOKEN="$HF_TOKEN" CUDA_VISIBLE_DEVICES=0 HF_USERNAME="$HF_USERNAME" python full_automation.py
else
    echo "数据处理失败，请检查日志文件。"
    exit 1
fi
EOF
    
    chmod +x run_training_node.sh
    
    # 使用 PM2 启动训练节点
    pm2 start run_training_node.sh --name "flock-training-node" -- start && pm2 save && pm2 startup
    
    echo "训练节点已启动。您可以使用 'pm2 logs flock-training-node' 查看日志。"
}

function update_task_id() {
    read -p "输入新的任务ID (TASK_ID): " NEW_TASK_ID
    
    # 更新验证者节点的 Task ID
    if [ -f "llm-loss-validator/run_validator.sh" ]; then
        sed -i "s/--task_id \".*\"/--task_id \"$NEW_TASK_ID\"/" llm-loss-validator/run_validator.sh
        pm2 restart llm-loss-validator
        echo "验证者节点的 Task ID 已更新并重启。"
    else
        echo "未找到验证者节点的运行脚本。"
    fi
    
    # 更新训练节点的 Task ID
    if [ -f "flk/run_training_node.sh" ]; then
        sed -i "s/TASK_ID=.*/TASK_ID=$NEW_TASK_ID/" flk/run_training_node.sh
        pm2 restart flock-training-node
        echo "训练节点的 Task ID 已更新并重启。"
    else
        echo "未找到训练节点的运行脚本。"
    fi
}

# 升级节点
function update_node() {
    # 升级验证者节点
    if [ -d "llm-loss-validator" ]; then
        cd llm-loss-validator && git pull && pm2 restart llm-loss-validator
        echo "验证者节点已升级."
    else
        echo "未找到验证者节点目录."
    fi

    # 升级训练节点
    if [ -d "flk" ]; then
        cd flk && git pull && pm2 restart flock-training-node
        echo "训练节点已升级."
    else
        echo "未找到训练节点目录."
    fi
}

# 主菜单
function main_menu() {
    clear
    echo "脚本以及教程由推特用户大赌哥 @y95277777 编写，免费开源，请勿相信收费"
    echo "=========================Flock节点安装======================================="
    echo "节点社区 Telegram 群组:https://t.me/niuwuriji"
    echo "节点社区 Telegram 频道:https://t.me/niuwuriji"
    echo "请选择要执行的操作:"
    echo "1. 安装验证者节点"
    echo "2. 安装训练节点"
    echo "3. 查看验证者节点日志"
    echo "4. 查看训练节点日志"
    echo "5. 删除常规节点"
    echo "6. 删除训练节点"
    echo "7. 修改任务 ID 并重启节点"
    echo "8. 升级节点"
    read -p "请输入选项（1-8）: " OPTION
    case $OPTION in
    1) install_node ;;
    2) install_train_node ;;
    3) check_node ;;
    4) pm2 logs flock-training-node ;;
    5) uninstall_node ;;
    6) pm2 delete flock-training-node && rm -rf flk ;;
    7) update_task_id ;;
    8) update_node ;;  # 添加升级节点功能
    *) echo "无效选项。" ;;
    esac
}

# 显示主菜单
main_menu