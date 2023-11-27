#!/bin/bash
echo '生成资源和老师统计表'
python excel_process.py
echo '生成资源评分'
python resource_scoring.py
echo '生成资源推荐列表'
python match.py
echo '根据用户信息输出生成10个推荐资源'
python query.py --user_ID 001e43e7931c4f3280dbee30a7164317