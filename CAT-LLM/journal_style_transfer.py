"""
Journal Style Transfer Script
基于CAT-LLM的实现方式,使用LLM进行学术期刊风格转换
支持OpenAI API和其他LLM接口
改进: 支持接受外部风格描述作为目标风格
"""

import json
import os
import time
import re
from typing import Dict, List, Optional
import requests


class JournalStyleTransfer:
    """学术期刊风格转换器"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_retries: int = 3,
        target_style: Optional[str] = None
    ):
        """
        初始化风格转换器

        Args:
            api_url: API地址
            api_key: API密钥
            model_name: 模型名称
            temperature: 温度参数
            max_retries: 最大重试次数
            target_style: 目标风格描述(如果为None,则使用默认风格描述)
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

        # 设置目标风格描述
        if target_style:
            self.target_style = target_style
            print("Using provided target style description")
        else:
            # 如果未提供,使用默认风格描述
            self.target_style = self._generate_default_target_style()
            print("Using default target style description")

    def set_target_style(self, style_description: str):
        """
        设置目标风格描述

        Args:
            style_description: 风格描述文本
        """
        self.target_style = style_description
        print("Target style description updated")

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

    def _generate_default_target_style(self) -> str:
        """
        生成目标期刊风格描述

        Returns:
            风格描述文本
        """
        # 使用之前分析的风格特征生成综合描述
        style_description = """This academic article follows the style of Information Processing & Management journal, characterized by:

From the sentence perspective:
- The average sentence length is 16-17 words, indicating balanced complexity suitable for academic discourse.
- Sentence length distribution shows a mix of short and medium sentences for clarity while maintaining analytical depth.
- The text employs passive voice selectively (10-15% of sentences) to maintain scholarly objectivity and emphasize actions rather than actors.

From the vocabulary perspective:
- The article uses 30-50 academic connectors (e.g., however, moreover, furthermore, therefore, consequently, specifically) to ensure strong logical flow and coherent argumentation.
- Academic verbs appear frequently (e.g., demonstrate, indicate, reveal, propose, investigate, analyze, evaluate, implement) showing substantial academic vocabulary for rigorous analysis.
- Academic adjectives occur regularly (e.g., significant, comprehensive, robust, novel, effective, systematic) reflecting appropriate use of evaluative language to emphasize research significance.

From the structural perspective:
- The article follows standard academic structure with sections: Abstract, Introduction, Related Work/Literature Review, Methods/Methodology, Results, Discussion, Conclusion, and References.
- The overall tone is objective, formal, and research-oriented, consistently avoiding colloquial expressions and maintaining scholarly rigor throughout.
- Citations and references are integrated systematically to support claims and arguments, following proper academic citation conventions.
- Technical terminology is used precisely and consistently, with clear definitions provided for domain-specific concepts.
- Arguments are evidence-based, supported by empirical data, theoretical frameworks, or prior research.
"""
        return style_description

    def transfer_text(self, source_text: str, retry_count: int = 0) -> Optional[str]:
        """
        将源文本转换为目标期刊风格

        Args:
            source_text: 源文本
            retry_count: 当前重试次数

        Returns:
            转换后的文本,失败返回None
        """
        # 构建提示词(参考CAT-LLM的提示词设计)
        prompt = f"""You are an expert academic writer specializing in Information Processing & Management journal style.

Please transform the following text into the academic style described below, while preserving the original meaning and content.

Original Text:
{source_text}

Target Style Description:
{self.target_style}

Requirements:
1. Maintain all the original information, data, and key points
2. Transform the writing style to match the target academic journal style
3. Use appropriate academic vocabulary and formal language
4. Ensure logical flow with proper connectors and transitions
5. Apply passive voice where appropriate for objectivity
6. Keep the technical accuracy and precision
7. Output the text as a single continuous paragraph WITHOUT line breaks or extra spaces between sentences
8. Only output the transformed text, without any explanations or meta-comments

Transformed Text:"""

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
            print(f"Error during style transfer (attempt {retry_count + 1}/{self.max_retries}): {e}")

            if retry_count < self.max_retries - 1:
                # 等待后重试
                time.sleep(5 * (retry_count + 1))
                return self.transfer_text(source_text, retry_count + 1)
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
    ) -> Dict[str, any]:
        """
        处理文件进行风格转换

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

        transformed_paragraphs = []
        success_count = 0
        fail_count = 0

        for i, paragraph in enumerate(paragraphs):
            print(f"Processing paragraph {i + 1}/{len(paragraphs)}...")

            # 如果段落太短,直接保留
            if len(paragraph.split()) < 10:
                transformed_paragraphs.append(paragraph)
                continue

            # 转换风格
            transformed = self.transfer_text(paragraph)

            if transformed:
                transformed_paragraphs.append(transformed)
                success_count += 1
            else:
                # 转换失败,保留原文
                transformed_paragraphs.append(paragraph)
                fail_count += 1

            # 保存中间结果
            if save_intermediate and (i + 1) % 5 == 0:
                self._save_intermediate_result(
                    output_file,
                    transformed_paragraphs,
                    i + 1,
                    len(paragraphs)
                )

            # 避免API频率限制
            time.sleep(1)

        # 合并转换后的段落
        final_text = '\n\n'.join(transformed_paragraphs)

        # 保存最终结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_text)

        print(f"\nStyle transfer completed!")
        print(f"Saved to: {output_file}")

        stats = {
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
        file_pattern: str = '*.txt'
    ):
        """
        批量处理文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            file_pattern: 文件匹配模式
        """
        import glob

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有匹配的文件
        pattern = os.path.join(input_dir, file_pattern)
        files = glob.glob(pattern)

        print(f"Found {len(files)} files to process")

        all_stats = []

        for input_file in files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(
                output_dir,
                filename.replace('.txt', '_journal_style.txt')
            )

            try:
                stats = self.process_file(input_file, output_file)
                stats['filename'] = filename
                all_stats.append(stats)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        # 保存总体统计
        stats_file = os.path.join(output_dir, 'transfer_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)

        print(f"\nBatch processing completed!")
        print(f"Statistics saved to: {stats_file}")


def main():
    """主函数"""
    # 配置API信息(需要用户提供)
    # 这里使用环境变量或配置文件读取
    api_url = os.getenv('OPENAI_API_URL', 'https://yunwu.zeabur.app/v1/chat/completions')
    api_key = os.getenv('OPENAI_API_KEY', '')

    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        return

    # 创建风格转换器
    transferer = JournalStyleTransfer(
        api_url=api_url,
        api_key=api_key,
        model_name='gpt-3.5-turbo',
        temperature=0.7
    )

    # 示例:处理单个文件
    # input_file = "path/to/your/input.txt"
    # output_file = "path/to/your/output.txt"
    # transferer.process_file(input_file, output_file)

    # 示例:批量处理
    # input_dir = "./input_texts"
    # output_dir = "./output_texts"
    # transferer.batch_process(input_dir, output_dir)

    print("Journal Style Transfer Tool initialized successfully!")
    print("Please use the transferer object to process your files.")


if __name__ == "__main__":
    main()
