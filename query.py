# 输入一个用户的ID，输出推荐的6个资源ID和资源名
import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_ID', default='001e43e7931c4f3280dbee30a7164317',
                        help='输入用户ID')

    opt = parser.parse_args()
    teacher_resources = pd.read_csv('teacher_resources.csv')
    resources = pd.read_csv('resource.csv')
    # 根据 opt.user_ID 在 teacher_resources 中查找对应的资源序号列表
    user_resources = teacher_resources[teacher_resources['用户id'] == opt.user_ID]['资源序号列表'].values
    # 输出列表中的前 10 个资源 ID
    if len(user_resources) > 0:
        resource_list = user_resources[0].split(',')[:10]  # 假设资源序号列表是以逗号分隔的字符串
        print("前10个资源ID:", resource_list)

        # 根据资源ID在resources表中查询对应的资源标题
        for resource_id in resource_list:
            resource_id = resource_id.replace("'", "").replace('"', '').replace(' ', '').replace('[', '').replace(']', '')
            resource_title = resources[resource_id == resources['资源ID']]['资源标题'].values
            if len(resource_title) > 0:
                print(f"资源ID为 {resource_id} 的资源标题为：{resource_title[0]}")
            else:
                print(f"未找到资源ID为 {resource_id} 的资源标题。")
    else:
        print("未找到匹配的资源序号列表。")

    pass

if __name__ == "__main__":
    main()