#!/bin/bash
###
 # @Author: wangschang wangschang@126.com
 # @Date: 2026-03-12 09:09:02
 # @LastEditors: wangschang wangschang@126.com
 # @LastEditTime: 2026-03-12 09:28:53
 # @FilePath: /CoPaw/start_app.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 
# 启动脚本：配置 LLM 和 Mem0 需要的环境变量并启动 CoPaw app

# 如果你的模型服务需要修改 URL 和 模型名字，请在这里修改
# 默认配置适用于兼容 OpenAI 格式的服务
# export OPENAI_API_KEY="sk-xuvpzwjkgfmzhajqtxolvhafawqxoqayjklciwzsabxvbndc"

# # （可选）配置代理地址或兼容服务的 Base URL
# export MODEL_BASE_URL="https://api.siliconflow.cn/v1"
# # （可选）配置具体的模型名
# export MODEL_NAME="Qwen/Qwen3-8B"

# （可选）是否开启 mem0 混合记忆，默认为 true
export MEM0_ENABLE="true"

echo "正在启动 CoPaw APP..."
python -m copaw app
