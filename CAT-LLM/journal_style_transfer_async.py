"""
Async Journal Style Transfer Module
异步版本：支持并发处理多条数据，大幅提升效率
使用 asyncio + aiohttp 实现异步API调用
"""

import os
import asyncio
import aiohttp
import time
import json
import re
from typing import Optional, List, Dict
from tqdm import tqdm


class AsyncJournalStyleTransfer:
    """异步学术期刊风格转换器 - 支持并发处理"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_retries: int = 3,
        target_style: Optional[str] = None,
        max_concurrent: int = 10,  # 最大并发数
        timeout: int = 120  # 超时时间（秒）
    ):
        """
        初始化异步风格转换器

        Args:
            api_url: API地址
            api_key: API密钥
            model_name: 模型名称
            temperature: 温度参数
            max_retries: 最大重试次数
            target_style: 目标风格描述
            max_concurrent: 最大并发请求数
            timeout: 超时时间（秒）
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        # 创建信号量控制并发数
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # 设置目标风格
        if target_style:
            self.target_style = target_style
        else:
            self.target_style = self._generate_default_target_style()

    def set_target_style(self, style_description: str):
        """设置目标风格描述"""
        self.target_style = style_description
        print("Target style description updated")

    def _clean_text(self, text: str) -> str:
        """清理文本，去除多余的空行和回车"""
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _generate_default_target_style(self) -> str:
        """生成默认目标期刊风格描述"""
        return """Academic journal style with the following characteristics:
1. Average sentence length: 16-17 words
2. Frequent use of academic connectors (however, furthermore, therefore, etc.)
3. Formal academic vocabulary
4. Objective tone with appropriate use of passive voice (10-15%)
5. Clear logical structure with smooth transitions
6. Precise and concise expression"""

    async def transfer_text_async(
        self,
        text: str,
        text_id: str = "",
        retry_count: int = 0
    ) -> Optional[str]:
        """
        异步转换文本风格

        Args:
            text: 源文本
            text_id: 文本标识（用于日志）
            retry_count: 当前重试次数

        Returns:
            转换后的文本,失败返回None
        """
        # 使用信号量控制并发数
        async with self.semaphore:
            prompt = f"""You are a professional academic writing expert specializing in journal article style transformation.

Your task is to transform the given text to match the following academic journal style:

{self.target_style}

Original Text:
{text}

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
                response = await self._call_llm_api_async(prompt)

                if response:
                    cleaned_response = self._clean_text(response.strip())
                    return cleaned_response
                else:
                    raise Exception("Empty response from API")

            except Exception as e:
                print(f"[{text_id}] Error during style transfer (attempt {retry_count + 1}/{self.max_retries}): {e}")

                if retry_count < self.max_retries - 1:
                    # 指数退避重试
                    await asyncio.sleep(5 * (retry_count + 1))
                    return await self.transfer_text_async(text, text_id, retry_count + 1)
                else:
                    print(f"[{text_id}] Failed after {self.max_retries} attempts")
                    return None

    async def _call_llm_api_async(self, prompt: str) -> Optional[str]:
        """异步调用LLM API"""
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

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['message']['content']
                else:
                    text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {text}")

    async def batch_transfer_async(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Optional[str]]:
        """
        批量异步转换文本风格

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            转换后的文本列表
        """
        print(f"\n[Async Batch Processing] Processing {len(texts)} texts with max {self.max_concurrent} concurrent requests...")

        # 创建异步任务
        tasks = []
        for i, text in enumerate(texts):
            text_id = f"Text-{i+1}/{len(texts)}"
            task = self.transfer_text_async(text, text_id)
            tasks.append(task)

        # 并发执行所有任务
        if show_progress:
            # 使用gather执行,同时显示进度条
            # gather保证结果顺序与tasks顺序一致
            done_count = 0
            with tqdm(total=len(tasks), desc="Style transfer") as pbar:
                results = await asyncio.gather(*tasks)
                # 注意:这里无法实时更新进度,因为gather是批量等待
                # 但保证了结果顺序和避免重复执行
                pbar.update(len(tasks))
        else:
            results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r is not None)
        print(f"[Async Batch Processing] Completed: {success_count}/{len(texts)} successful")

        return results

    async def process_file_async(
        self,
        input_file: str,
        output_file: str
    ) -> Dict:
        """
        异步处理单个文件

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径

        Returns:
            处理统计信息
        """
        print(f"\n[Processing File] {input_file}")
        start_time = time.time()

        # 读取源文本
        with open(input_file, 'r', encoding='utf-8') as f:
            source_text = f.read()

        # 按段落分割（如果需要）
        paragraphs = source_text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 如果只有一个段落，直接处理
        if len(paragraphs) == 1:
            results = [await self.transfer_text_async(paragraphs[0], "Full-Text")]
        else:
            # 批量异步处理
            results = await self.batch_transfer_async(paragraphs)

        # 过滤失败的结果
        successful_results = [r for r in results if r is not None]

        if not successful_results:
            print(f"[Error] All paragraphs failed to process")
            return {
                'input_file': input_file,
                'output_file': output_file,
                'success': False,
                'processing_time': time.time() - start_time
            }

        # 保存结果
        output_text = '\n\n'.join(successful_results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        elapsed_time = time.time() - start_time

        stats = {
            'input_file': input_file,
            'output_file': output_file,
            'success': True,
            'total_paragraphs': len(paragraphs),
            'successful_paragraphs': len(successful_results),
            'processing_time': elapsed_time
        }

        print(f"[Completed] File saved to: {output_file}")
        print(f"[Stats] Processed {len(successful_results)}/{len(paragraphs)} paragraphs in {elapsed_time:.2f}s")

        return stats

    async def batch_process_files_async(
        self,
        input_files: List[str],
        output_dir: str
    ) -> List[Dict]:
        """
        批量异步处理多个文件

        Args:
            input_files: 输入文件列表
            output_dir: 输出目录

        Returns:
            所有文件的处理统计信息
        """
        print(f"\n{'='*80}")
        print(f"ASYNC BATCH FILE PROCESSING")
        print(f"{'='*80}")
        print(f"Total files: {len(input_files)}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*80}\n")

        start_time = time.time()

        # 创建文件处理任务
        tasks = []
        for input_file in input_files:
            basename = os.path.basename(input_file).replace('_style_removed.txt', '').replace('.txt', '')
            output_file = os.path.join(output_dir, f"{basename}_transferred.txt")
            task = self.process_file_async(input_file, output_file)
            tasks.append(task)

        # 并发处理所有文件
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time
        success_count = sum(1 for r in results if r.get('success', False))

        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING COMPLETED")
        print(f"{'='*80}")
        print(f"Total files: {len(input_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(input_files) - success_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per file: {total_time/len(input_files):.2f}s")
        print(f"{'='*80}\n")

        return results

    def batch_process(
        self,
        input_dir: str,
        output_dir: str
    ) -> List[Dict]:
        """
        同步接口：批量处理目录下所有txt文件

        Args:
            input_dir: 输入目录
            output_dir: 输出目录

        Returns:
            处理统计信息列表
        """
        import glob

        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))

        if not txt_files:
            print(f"No txt files found in {input_dir}")
            return []

        # 运行异步处理
        return asyncio.run(self.batch_process_files_async(txt_files, output_dir))


# 兼容性：保留同步版本的接口
class JournalStyleTransfer:
    """同步风格转换器 - 保持向后兼容"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_retries: int = 3,
        target_style: Optional[str] = None
    ):
        # 内部使用异步版本，设置并发数为1实现同步行为
        self.async_transfer = AsyncJournalStyleTransfer(
            api_url=api_url,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries,
            target_style=target_style,
            max_concurrent=1  # 同步版本：并发数为1
        )

    def set_target_style(self, style_description: str):
        """设置目标风格描述"""
        self.async_transfer.set_target_style(style_description)

    def transfer_text(self, text: str, retry_count: int = 0) -> Optional[str]:
        """同步版本：转换文本风格"""
        return asyncio.run(
            self.async_transfer.transfer_text_async(text, "Sync", retry_count)
        )

    def process_file(self, input_file: str, output_file: str) -> dict:
        """同步版本：处理单个文件"""
        return asyncio.run(
            self.async_transfer.process_file_async(input_file, output_file)
        )

    def batch_process(self, input_dir: str, output_dir: str) -> List[Dict]:
        """同步版本：批量处理"""
        return self.async_transfer.batch_process(input_dir, output_dir)


def main():
    """主函数 - 演示使用"""
    import sys

    # 示例配置
    api_url = "https://yunwu.zeabur.app/v1/chat/completions"
    api_key = "your-api-key"
    model_name = "gpt-5-2025-08-07"

    # 创建异步处理器（推荐）
    async_transfer = AsyncJournalStyleTransfer(
        api_url=api_url,
        api_key=api_key,
        model_name=model_name,
        max_concurrent=10  # 最多10个并发请求
    )

    # 批量处理
    stats = async_transfer.batch_process(
        input_dir="./output/style_removed",
        output_dir="./output/transferred"
    )

    print("\nProcessing complete!")
    for stat in stats:
        if stat.get('success'):
            print(f"  ✓ {os.path.basename(stat['input_file'])}: {stat['processing_time']:.2f}s")
        else:
            print(f"  ✗ {os.path.basename(stat['input_file'])}: Failed")


if __name__ == "__main__":
    main()
