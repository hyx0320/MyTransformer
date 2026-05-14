import subprocess
import sys
import os

# 设置环境变量避免 GBK 终端下 emoji 报错
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

def run_script(script_name):
    print(f"\n===== 正在运行：{script_name} =====")
    # 用当前 Python 环境运行脚本
    result = subprocess.run([sys.executable, script_name])
    
    # 如果运行失败，直接停止，不继续往下跑
    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} 运行失败！")
        sys.exit(1)

if __name__ == "__main__":
    # 按你想要的顺序写在这里
    run_script("train.py")
    run_script("split_checkpoint.py")
    run_script("test.py")
    run_script("analysis.py")
    run_script("generate_report.py")

    print("\n[OK] 所有脚本全部顺序执行完成！")