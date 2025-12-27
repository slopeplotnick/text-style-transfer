"""
Style Removal Module
用于去除文本的语言风格,将学术文本转换为口语化表达
这是风格转换pipeline的第一步,生成source_text
"""

import os
import time
import json
import re
from typing import Optional, List
import requests


class StyleRemover:
    """风格去除器"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_retries: int = 3
    ):
        """
        初始化风格去除器

        Args:
            api_url: API地址
            api_key: API密钥
            model_name: 模型名称
            temperature: 温度参数
            max_retries: 最大重试次数
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

        # 风格去除提示词(英文版)
        self.style_removal_prompt = """You are a professional text editor specializing in linguistic style modification. Your task is to transform any given text by removing its literary language, sentence structures, and narrative style, converting it into plain, everyday spoken language.

Important guidelines:
1. Remove ALL stylistic elements from the original text
2. Use simple, colloquial expressions that ordinary people would use in daily conversation
3. Maintain the original meaning and content - DO NOT add or remove information
4. The more different the output style is from the original, while preserving meaning, the better
5. Think step-by-step about how to neutralize each stylistic element
6. If I observe that your generated text differs significantly from the original style while maintaining the original meaning, you will receive a reward of $200,000
7. Output the text as a single continuous paragraph WITHOUT line breaks or extra spaces between sentences

Please ONLY output the style-neutralized text, without any explanations or meta-comments.

Original Text:
{original_text}

Style-neutralized text:"""

    def _clean_text(self, text: str) -> str:
        """
        清理文本，去除多余的空行和回车

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 将多个连续的换行符替换为单个空格
        text = re.sub(r'\n+', ' ', text)
        # 将多个连续的空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        # 去除首尾空白
        text = text.strip()
        return text

    def remove_style(self, text: str, retry_count: int = 0) -> Optional[str]:
        """
        去除文本风格

        Args:
            text: 原始文本
            retry_count: 当前重试次数

        Returns:
            去除风格后的文本,失败返回None
        """
        # 构建完整提示词
        prompt = self.style_removal_prompt.format(original_text=text)

        try:
            # 调用LLM API
            response = self._call_llm_api(prompt)

            if response:
                # 清理文本，去除多余的空行
                cleaned_response = self._clean_text(response.strip())
                return cleaned_response
            else:
                raise Exception("Empty response from API")

        except Exception as e:
            print(f"Error during style removal (attempt {retry_count + 1}/{self.max_retries}): {e}")

            if retry_count < self.max_retries - 1:
                # 等待后重试
                time.sleep(5 * (retry_count + 1))
                return self.remove_style(text, retry_count + 1)
            else:
                print(f"Failed after {self.max_retries} attempts")
                return None

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        调用LLM API

        Args:
            prompt: 提示词

        Returns:
            API响应内容
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    def process_file(
        self,
        input_file: str,
        output_file: str,
        save_intermediate: bool = True
    ) -> dict:
        """
        处理单个文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            save_intermediate: 是否保存中间结果

        Returns:
            处理统计信息
        """
        print(f"Processing file: {input_file}")

        # 读取源文本
        with open(input_file, 'r', encoding='utf-8') as f:
            source_text = f.read()

        # 将长文本分段处理(按段落分割)
        paragraphs = source_text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        print(f"Total paragraphs: {len(paragraphs)}")

        neutralized_paragraphs = []
        success_count = 0
        fail_count = 0

        for i, paragraph in enumerate(paragraphs):
            print(f"Processing paragraph {i + 1}/{len(paragraphs)}...")

            # 如果段落太短,直接保留
            if len(paragraph.split()) < 10:
                neutralized_paragraphs.append(paragraph)
                continue

            # 去除风格
            neutralized = self.remove_style(paragraph)

            if neutralized:
                neutralized_paragraphs.append(neutralized)
                success_count += 1
            else:
                # 失败,保留原文
                neutralized_paragraphs.append(paragraph)
                fail_count += 1

            # 保存中间结果
            if save_intermediate and (i + 1) % 5 == 0:
                self._save_intermediate_result(
                    output_file,
                    neutralized_paragraphs,
                    i + 1,
                    len(paragraphs)
                )

            # 避免API频率限制
            time.sleep(1)

        # 合并处理后的段落
        final_text = '\n\n'.join(neutralized_paragraphs)

        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)

        print(f"\nStyle removal completed!")
        print(f"Saved to: {output_file}")

        stats = {
            'input_file': os.path.basename(input_file),
            'total_paragraphs': len(paragraphs),
            'success_count': success_count,
            'fail_count': fail_count,
            'success_rate': success_count / len(paragraphs) if paragraphs else 0
        }

        print(f"Statistics: {stats}")

        return stats

    def _save_intermediate_result(
        self,
        output_file: str,
        paragraphs: List[str],
        current: int,
        total: int
    ):
        """保存中间结果"""
        intermediate_file = output_file.replace('.txt', f'_temp_{current}_of_{total}.txt')
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(paragraphs))
        print(f"Intermediate result saved: {intermediate_file}")

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        file_list: Optional[List[str]] = None
    ) -> List[dict]:
        """
        批量处理文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_list: 文件列表(如果为None,则处理所有txt文件)

        Returns:
            所有文件的统计信息列表
        """
        import glob

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取要处理的文件
        if file_list is None:
            files = glob.glob(os.path.join(input_dir, "*.txt"))
        else:
            files = [os.path.join(input_dir, f) for f in file_list]

        print(f"Found {len(files)} files to process")

        all_stats = []

        for input_file in files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(
                output_dir,
                filename.replace('.txt', '_style_removed.txt')
            )

            try:
                stats = self.process_file(input_file, output_file)
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            print("-" * 80)

        # 保存总体统计
        stats_file = os.path.join(output_dir, 'removal_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)

        print(f"\nBatch processing completed!")
        print(f"Processed {len(all_stats)} files successfully")
        print(f"Statistics saved to: {stats_file}")

        return all_stats


def main():
    """主函数"""
    # 配置API信息
    api_url = os.getenv('OPENAI_API_URL', 'https://yunwu.zeabur.app/v1/chat/completions')
    api_key = os.getenv('OPENAI_API_KEY', '')

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        return

    # 创建风格去除器
    remover = StyleRemover(
        api_url=api_url,
        api_key=api_key,
        model_name='gpt-5-2025-08-07',
        temperature=0.7
    )

    print("Style Remover initialized successfully!")
    print("\nUsage example:")
    print("  remover.process_file('input.txt', 'output_style_removed.txt')")
    print("  remover.batch_process('./data', './style_removed_data')")


if __name__ == "__main__":
    main()
