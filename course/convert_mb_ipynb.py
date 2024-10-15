import os
import re
import subprocess
import tempfile
from tqdm import tqdm

# ## 转换 notedown

# ```
# notedown transformer.md > transformer.ipynb
# ```

def process_md_content(content):
    # 定义正则表达式模式
    pattern = re.compile(r"```{\.python \.input}\n(.*?)```", re.DOTALL)

    def keep_or_remove(match):
        code_block = match.group(1)
        first_line = code_block.split("\n", 1)[0]
        if first_line.startswith("#@tab all") or re.search(
            r"#@tab.*pytorch", first_line
        ):
            return match.group(0)  # 保留整个代码块
        else:
            return ""  # 删除代码块

    # 使用正则表达式处理内容
    processed_content = pattern.sub(keep_or_remove, content)

    # 替换 from d2l import torch as d2l 为 import d2l
    processed_content = processed_content.replace(
        "from d2l import torch as d2l", "import d2l"
    )

    # 删除 :width:`500px` 和 :height:`500px` 代表宽或高的字符串
    processed_content = re.sub(r":width:`\d+px`", "", processed_content)
    processed_content = re.sub(r":height:`\d+px`", "", processed_content)

    # 将 :label: 和 :numref: 前部分替换为 【】
    processed_content = re.sub(r":label:`(.*?)`", r"【\1】", processed_content)
    processed_content = re.sub(r":numref:`(.*?)`", r"【\1】", processed_content)

    # 只保留 :begin_tab:`pytorch` 块内代码
    processed_content = re.sub(
        r":begin_tab:`pytorch`(.*?)\n:end_tab:",
        r"\1",
        processed_content,
        flags=re.DOTALL,
    )
    processed_content = re.sub(
        r":begin_tab:`.*?`\n.*?\n:end_tab:", "", processed_content, flags=re.DOTALL
    )

    return processed_content


def convert_md_to_ipynb():
    # 递归查找所有 .md 文件，但排除以 _origin.md 结尾的文件
    md_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".md") and not file.endswith("_origin.md"):
                md_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    pbar = tqdm(md_files, desc="Processing files", unit="file")
    for md_file in pbar:
        ipynb_file = os.path.splitext(md_file)[0] + ".ipynb"

        # 更新进度条描述为当前文件名
        pbar.set_description(f"Processing {md_file}")

        # 读取 .md 文件内容
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 使用专门的函数处理内容
        processed_content = process_md_content(content)

        # 创建临时文件保存处理后的内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as temp_md_file:
            temp_md_file.write(processed_content.encode("utf-8"))
            temp_md_file_path = temp_md_file.name

        # 执行 notedown 命令进行转换
        subprocess.run(
            ["notedown", temp_md_file_path, "--to", "notebook", ">", ipynb_file],
            shell=True,
        )

        # 删除临时文件
        os.remove(temp_md_file_path)


if __name__ == "__main__":
    convert_md_to_ipynb()
