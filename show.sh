#!/bin/bash

# 1. 获取当前终端的总行数
TOTAL_ROWS=$(tput lines)

# 2. 计算滚动区域：从第 1 行到倒数第 5 行
# 例如 40 行的终端，区域设为 1-35
SCROLL_END=$((TOTAL_ROWS - 5))

# 3. 设置滚动区域 (DECSTBM)
# \033[top;bottomr
printf "\033[1;${SCROLL_END}r"

# 4. 清屏并将光标置于顶部，开始你的演示
clear
# 这里运行你的程序，同时强制实时输出
stdbuf -o L bash run_sparkinfer.sh release cli

# 5. 【重要】演示结束后恢复全屏滚动，否则终端会很难用
trap "printf '\033[r'; clear" EXIT
