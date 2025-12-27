"""
NLTK资源下载脚本
自动下载项目所需的所有NLTK资源
"""

import nltk
import sys

def download_nltk_resources():
    """下载所有需要的NLTK资源"""

    print("开始下载NLTK资源...")
    print("=" * 60)

    resources = [
        'punkt',        # 句子分词器
        # 'punkt_tab',    # 句子分词器(新版)
        # 'averaged_perceptron_tagger',  # 词性标注
        # 'wordnet',      # 词典
        # 'stopwords',    # 停用词
    ]

    success_count = 0
    fail_count = 0

    for resource in resources:
        try:
            print(f"\n下载 {resource}...", end=" ")
            nltk.download(resource)
            print("✓ 成功")
            success_count += 1
        except Exception as e:
            print(f"✗ 失败: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"下载完成: 成功 {success_count}, 失败 {fail_count}")

    if fail_count > 0:
        print("\n⚠️ 部分资源下载失败，但不影响核心功能")
    else:
        print("\n✓ 所有资源下载成功！")

    return fail_count == 0

if __name__ == "__main__":
    try:
        success = download_nltk_resources()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 下载过程出错: {e}")
        sys.exit(1)
