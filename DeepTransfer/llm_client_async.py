"""
Async LLM Client for DeepTransfer
使用 asyncio + aiohttp 实现异步 API 调用
"""
import asyncio
import aiohttp
from typing import Optional


class AsyncLLMClient:
    """
    异步 LLM 客户端，支持并发 API 调用
    """
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        max_concurrent: int = 10,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        初始化异步 LLM 客户端

        Args:
            api_url: API URL
            api_key: API 密钥
            model: 模型名称
            max_concurrent: 最大并发请求数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # 创建信号量控制并发数
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_async(
        self,
        prompt: str,
        temperature: float = 0.3,
        request_id: str = "",
        retry_count: int = 0
    ) -> Optional[str]:
        """
        异步生成文本

        Args:
            prompt: 提示词
            temperature: 温度参数
            request_id: 请求标识（用于日志）
            retry_count: 当前重试次数

        Returns:
            生成的文本，失败返回 None
        """
        # 使用信号量控制并发数
        async with self.semaphore:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }

            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.api_url,
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result['choices'][0]['message']['content']
                        else:
                            text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {text}")

            except Exception as e:
                if request_id:
                    print(f"[{request_id}] LLM Call Error (attempt {retry_count + 1}/{self.max_retries}): {e}")
                else:
                    print(f"LLM Call Error (attempt {retry_count + 1}/{self.max_retries}): {e}")

                # 重试逻辑
                if retry_count < self.max_retries - 1:
                    # 指数退避
                    await asyncio.sleep(2 ** retry_count)
                    return await self.generate_async(prompt, temperature, request_id, retry_count + 1)
                else:
                    if request_id:
                        print(f"[{request_id}] Failed after {self.max_retries} attempts")
                    return None

    async def batch_generate_async(
        self,
        prompts: list,
        temperature: float = 0.3,
        show_progress: bool = True
    ) -> list:
        """
        批量异步生成文本

        Args:
            prompts: 提示词列表
            temperature: 温度参数
            show_progress: 是否显示进度

        Returns:
            生成的文本列表
        """
        print(f"\n[Async Batch Generation] Processing {len(prompts)} requests...")

        # 创建异步任务
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = f"Request-{i+1}/{len(prompts)}"
            task = self.generate_async(prompt, temperature, request_id)
            tasks.append(task)

        # 并发执行所有任务
        results = await asyncio.gather(*tasks)

        success_count = sum(1 for r in results if r is not None)
        print(f"[Async Batch Generation] Completed: {success_count}/{len(prompts)} successful")

        return results


# 同步包装器（保持向后兼容）
class LLMClient:
    """
    同步 LLM 客户端（使用异步客户端实现）
    """
    def __init__(self, api_url: str, api_key: str, model: str):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

        # 内部使用异步客户端，并发数为 1
        self.async_client = AsyncLLMClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            max_concurrent=1
        )

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        同步生成文本
        """
        result = asyncio.run(
            self.async_client.generate_async(prompt, temperature)
        )

        if result is None:
            return f"[Error generating text]"

        return result
