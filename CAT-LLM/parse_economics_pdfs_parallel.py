#!/usr/bin/env python3
"""
批量调用MinerU解析Economics目录下的所有PDF文件（并行版本）
服务器路径: /root/datasets/text_style_transfer/Economics
"""

import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# 配置日志
def setup_logger(log_file=None):
    """设置日志"""
    if log_file is None:
        log_file = f'mineru_parse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class MinerUParser:
    """MinerU PDF解析器"""

    def __init__(self, base_dir: str, output_dir: str, workers: int = 4,
                 skip_existing: bool = True, timeout: int = 300):
        """
        初始化解析器

        Args:
            base_dir: 源PDF目录
            output_dir: 输出目录
            workers: 并行工作线程数
            skip_existing: 是否跳过已解析的文件
            timeout: 单个文件解析超时时间（秒）
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.workers = workers
        self.skip_existing = skip_existing
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def find_all_pdfs(self) -> List[Tuple[str, str]]:
        """递归查找所有PDF文件"""
        pdf_files = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    rel_path = os.path.relpath(pdf_path, self.base_dir)
                    pdf_files.append((pdf_path, rel_path))
        return sorted(pdf_files)

    def is_already_parsed(self, pdf_path: str, rel_path: str) -> bool:
        """检查文件是否已被解析"""
        if not self.skip_existing:
            return False

        output_dir = os.path.join(self.output_dir, os.path.dirname(rel_path))
        output_base = os.path.splitext(os.path.basename(pdf_path))[0]

        # 检查可能的输出文件格式
        possible_outputs = [
            os.path.join(output_dir, f"{output_base}.md"),
            os.path.join(output_dir, f"{output_base}.json"),
            os.path.join(output_dir, output_base, "result.md"),
            os.path.join(output_dir, output_base, "result.json"),
        ]

        return any(os.path.exists(f) for f in possible_outputs)

    def parse_single_pdf(self, pdf_path: str, rel_path: str) -> Tuple[bool, str, str]:
        """
        解析单个PDF文件

        Args:
            pdf_path: PDF文件路径
            rel_path: 相对路径

        Returns:
            (成功标志, PDF路径, 错误信息)
        """
        try:
            # 检查是否已解析
            if self.is_already_parsed(pdf_path, rel_path):
                self.logger.info(f"已存在解析结果，跳过: {rel_path}")
                return True, pdf_path, "已存在"

            # 构造输出路径
            output_subdir = os.path.join(self.output_dir, os.path.dirname(rel_path))
            os.makedirs(output_subdir, exist_ok=True)

            # 调用MinerU命令
            # 根据实际MinerU部署调整命令格式
            # 常见格式: magic-pdf -i input.pdf -o output_dir
            # 或: magic-pdf --pdf input.pdf --output-dir output_dir
            cmd = [
                "magic-pdf",
                "-i", pdf_path,
                "-o", output_subdir
            ]

            self.logger.info(f"开始解析: {rel_path}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode == 0:
                self.logger.info(f"✓ 解析成功: {rel_path}")
                return True, pdf_path, None
            else:
                error_msg = result.stderr or result.stdout or "未知错误"
                self.logger.error(f"✗ 解析失败: {rel_path}")
                self.logger.error(f"  错误信息: {error_msg[:200]}")
                return False, pdf_path, error_msg

        except subprocess.TimeoutExpired:
            error_msg = f"解析超时（>{self.timeout}秒）"
            self.logger.error(f"✗ {error_msg}: {rel_path}")
            return False, pdf_path, error_msg
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"✗ 解析异常: {rel_path}, 错误: {error_msg}")
            return False, pdf_path, error_msg

    def parse_all(self):
        """批量解析所有PDF"""
        self.logger.info("=" * 80)
        self.logger.info("开始批量解析Economics目录下的PDF文件")
        self.logger.info(f"源目录: {self.base_dir}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"并行线程数: {self.workers}")
        self.logger.info(f"超时时间: {self.timeout}秒")
        self.logger.info("=" * 80)

        # 查找所有PDF
        self.logger.info("正在搜索PDF文件...")
        pdf_files = self.find_all_pdfs()
        total_count = len(pdf_files)
        self.logger.info(f"找到 {total_count} 个PDF文件\n")

        if total_count == 0:
            self.logger.warning("未找到PDF文件，退出")
            return

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 统计信息
        success_count = 0
        fail_count = 0
        failed_files = []

        # 并行解析
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # 提交所有任务
            future_to_pdf = {
                executor.submit(self.parse_single_pdf, pdf_path, rel_path): (pdf_path, rel_path)
                for pdf_path, rel_path in pdf_files
            }

            # 处理完成的任务
            for idx, future in enumerate(as_completed(future_to_pdf), 1):
                pdf_path, rel_path = future_to_pdf[future]

                try:
                    success, _, error_msg = future.result()

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_files.append({
                            "file": rel_path,
                            "error": error_msg
                        })

                    # 显示进度
                    if idx % 10 == 0 or idx == total_count:
                        self.logger.info(
                            f"\n进度: [{idx}/{total_count}] "
                            f"成功: {success_count}, 失败: {fail_count}"
                        )

                except Exception as e:
                    self.logger.error(f"处理任务异常: {rel_path}, 错误: {str(e)}")
                    fail_count += 1
                    failed_files.append({
                        "file": rel_path,
                        "error": str(e)
                    })

        # 输出统计信息
        self.logger.info("\n" + "=" * 80)
        self.logger.info("解析完成统计:")
        self.logger.info(f"总文件数: {total_count}")
        self.logger.info(f"成功: {success_count}")
        self.logger.info(f"失败: {fail_count}")
        if total_count > 0:
            self.logger.info(f"成功率: {success_count/total_count*100:.2f}%")

        # 保存失败文件列表
        if failed_files:
            self.logger.info(f"\n失败文件数: {len(failed_files)}")
            failed_list_file = f'failed_files_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(failed_list_file, 'w', encoding='utf-8') as f:
                json.dump(failed_files, f, ensure_ascii=False, indent=2)
            self.logger.info(f"失败文件列表已保存到: {failed_list_file}")

        self.logger.info("=" * 80)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='批量调用MinerU解析Economics目录下的PDF文件'
    )
    parser.add_argument(
        '--base-dir',
        default='/root/datasets/text_style_transfer/Economics',
        help='源PDF目录路径'
    )
    parser.add_argument(
        '--output-dir',
        default='/root/datasets/text_style_transfer/Economics_parsed',
        help='输出目录路径'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='并行工作线程数 (默认: 4)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='单个文件解析超时时间(秒) (默认: 300)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='不跳过已解析的文件，重新解析所有文件'
    )
    parser.add_argument(
        '--log-file',
        help='日志文件路径'
    )

    args = parser.parse_args()

    # 设置日志
    setup_logger(args.log_file)

    # 创建解析器并执行
    parser_instance = MinerUParser(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        workers=args.workers,
        skip_existing=not args.no_skip_existing,
        timeout=args.timeout
    )

    parser_instance.parse_all()

if __name__ == "__main__":
    main()
