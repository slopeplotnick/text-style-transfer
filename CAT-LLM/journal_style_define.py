"""
Journal Style Definition Module
基于Information Processing & Management期刊风格特征定义模块
参考CAT-LLM的风格定义方法,为学术期刊文章定义风格特征
改进: 支持对整个data目录进行批量分析并生成聚合风格描述
"""

import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import os
import glob


class JournalStyleAnalyzer:
    """学术期刊风格分析器"""

    def __init__(self):
        """初始化分析器"""
        # 学术期刊常用连接词
        self.academic_connectors = {
            'however', 'moreover', 'furthermore', 'therefore', 'thus',
            'consequently', 'additionally', 'specifically', 'particularly',
            'notably', 'significantly', 'accordingly', 'meanwhile',
            'subsequently', 'conversely', 'nonetheless', 'nevertheless'
        }

        # 学术期刊常用动词(被动语态标记)
        self.academic_verbs = {
            'demonstrate', 'indicate', 'reveal', 'show', 'suggest',
            'propose', 'present', 'introduce', 'investigate', 'analyze',
            'examine', 'evaluate', 'assess', 'validate', 'implement',
            'conduct', 'perform', 'achieve', 'obtain', 'utilize'
        }

        # 学术期刊常用形容词
        self.academic_adjectives = {
            'significant', 'comprehensive', 'systematic', 'robust',
            'effective', 'efficient', 'novel', 'innovative', 'critical',
            'substantial', 'considerable', 'extensive', 'rigorous'
        }

    def read_file(self, file_path: str) -> str:
        """
        读取文件内容

        Args:
            file_path: 文件路径

        Returns:
            处理后的文本内容
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def extract_sentences(self, content: str) -> List[str]:
        """
        提取文本中的句子

        Args:
            content: 文本内容

        Returns:
            句子列表
        """
        # 使用正则表达式分割句子(以句号、问号、感叹号为分隔符)
        sentences = re.split(r'[.!?]\s+', content)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def count_sentences(self, sentences: List[str]) -> int:
        """统计句子总数"""
        return len([s for s in sentences if s.strip()])

    def count_words(self, text: str) -> int:
        """
        统计单词总数(英文文本)

        Args:
            text: 文本内容

        Returns:
            单词数量
        """
        # 移除标点符号,只保留字母和数字
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        words = cleaned_text.split()
        return len([w for w in words if w.strip()])

    def calculate_average_sentence_length(self, text: str, sentence_count: int) -> float:
        """
        计算平均句长

        Args:
            text: 文本内容
            sentence_count: 句子数量

        Returns:
            平均句长
        """
        word_count = self.count_words(text)
        if sentence_count == 0:
            return 0
        return round(word_count / sentence_count, 2)

    def analyze_sentence_length_distribution(self, sentences: List[str]) -> Dict[str, float]:
        """
        分析句子长度分布

        Args:
            sentences: 句子列表

        Returns:
            长度分布字典
        """
        short = 0  # 0-15 words
        medium = 0  # 16-25 words
        long = 0  # 26-35 words
        very_long = 0  # 35+ words

        for sentence in sentences:
            word_count = self.count_words(sentence)
            if word_count <= 15:
                short += 1
            elif word_count <= 25:
                medium += 1
            elif word_count <= 35:
                long += 1
            else:
                very_long += 1

        total = len(sentences)
        if total == 0:
            return {}

        return {
            'short': round(short / total, 3),
            'medium': round(medium / total, 3),
            'long': round(long / total, 3),
            'very_long': round(very_long / total, 3)
        }

    def analyze_academic_vocabulary(self, text: str) -> Dict[str, int]:
        """
        分析学术词汇使用情况

        Args:
            text: 文本内容

        Returns:
            学术词汇统计
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        connector_count = sum(1 for w in words if w in self.academic_connectors)
        verb_count = sum(1 for w in words if w in self.academic_verbs)
        adjective_count = sum(1 for w in words if w in self.academic_adjectives)

        return {
            'academic_connectors': connector_count,
            'academic_verbs': verb_count,
            'academic_adjectives': adjective_count
        }

    def detect_passive_voice(self, text: str) -> int:
        """
        检测被动语态使用次数

        Args:
            text: 文本内容

        Returns:
            被动语态出现次数
        """
        # 简单的被动语态检测: be动词 + 过去分词
        passive_patterns = [
            r'\b(is|are|was|were|been|be)\s+\w+ed\b',
            r'\b(is|are|was|were|been|be)\s+\w+en\b'
        ]

        count = 0
        for pattern in passive_patterns:
            matches = re.findall(pattern, text.lower())
            count += len(matches)

        return count

    def analyze_structure(self, text: str) -> Dict[str, bool]:
        """
        分析文章结构完整性

        Args:
            text: 文本内容

        Returns:
            结构完整性字典
        """
        text_lower = text.lower()

        # 检测标准学术论文章节
        has_abstract = bool(re.search(r'\babstract\b', text_lower))
        has_introduction = bool(re.search(r'\bintroduction\b', text_lower))
        has_method = bool(re.search(r'\b(method|methodology|approach)\b', text_lower))
        has_results = bool(re.search(r'\bresults?\b', text_lower))
        has_discussion = bool(re.search(r'\bdiscussion\b', text_lower))
        has_conclusion = bool(re.search(r'\bconclusion\b', text_lower))
        has_references = bool(re.search(r'\b(reference|bibliography)\b', text_lower))

        return {
            'has_abstract': has_abstract,
            'has_introduction': has_introduction,
            'has_method': has_method,
            'has_results': has_results,
            'has_discussion': has_discussion,
            'has_conclusion': has_conclusion,
            'has_references': has_references
        }

    def analyze_all_texts(self, data_dir: str) -> str:
        """
        分析data目录下所有文本并生成综合风格描述

        Args:
            data_dir: 数据目录路径

        Returns:
            综合风格描述字符串
        """
        # 获取所有txt文件
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))

        if not txt_files:
            raise ValueError(f"No .txt files found in {data_dir}")

        print(f"Found {len(txt_files)} journal articles to analyze")

        # 收集所有统计信息
        all_avg_lengths = []
        all_length_dists = []
        all_academic_vocabs = []
        all_passive_counts = []
        all_sentence_counts = []
        all_structures = []

        for txt_file in txt_files:
            print(f"Analyzing: {os.path.basename(txt_file)}")

            try:
                # 读取文件
                content = self.read_file(txt_file)

                # 提取句子
                sentences = self.extract_sentences(content)
                sentence_count = self.count_sentences(sentences)

                if sentence_count == 0:
                    print(f"  Warning: No sentences found, skipping")
                    continue

                # 计算各项指标
                avg_length = self.calculate_average_sentence_length(content, sentence_count)
                length_dist = self.analyze_sentence_length_distribution(sentences)
                academic_vocab = self.analyze_academic_vocabulary(content)
                passive_count = self.detect_passive_voice(content)
                structure = self.analyze_structure(content)

                # 收集统计信息
                all_avg_lengths.append(avg_length)
                all_length_dists.append(length_dist)
                all_academic_vocabs.append(academic_vocab)
                all_passive_counts.append(passive_count)
                all_sentence_counts.append(sentence_count)
                all_structures.append(structure)

                print(f"  Avg sentence length: {avg_length}, Sentences: {sentence_count}")

            except Exception as e:
                print(f"  Error analyzing {txt_file}: {e}")
                continue

        if not all_avg_lengths:
            raise ValueError("No valid articles could be analyzed")

        # 计算平均值
        avg_of_avg_lengths = sum(all_avg_lengths) / len(all_avg_lengths)

        # 聚合长度分布
        aggregated_length_dist = self._aggregate_length_distributions(all_length_dists)

        # 聚合学术词汇
        aggregated_vocab = self._aggregate_academic_vocab(all_academic_vocabs)

        # 计算平均被动语态数量
        avg_passive_count = sum(all_passive_counts) / len(all_passive_counts)
        avg_sentence_count = sum(all_sentence_counts) / len(all_sentence_counts)

        # 聚合结构信息
        aggregated_structure = self._aggregate_structures(all_structures)

        print(f"\nAnalyzed {len(all_avg_lengths)} articles successfully")
        print(f"Average sentence length across all articles: {avg_of_avg_lengths:.2f}")

        # 生成综合风格描述
        style_desc = self._format_aggregated_style_description(
            avg_of_avg_lengths,
            aggregated_length_dist,
            aggregated_vocab,
            avg_passive_count,
            aggregated_structure,
            avg_sentence_count
        )

        return style_desc

    def _aggregate_length_distributions(self, dists: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合长度分布"""
        if not dists:
            return {}

        keys = dists[0].keys()
        aggregated = {}
        for key in keys:
            aggregated[key] = sum(d.get(key, 0) for d in dists) / len(dists)

        return aggregated

    def _aggregate_academic_vocab(self, vocabs: List[Dict[str, int]]) -> Dict[str, float]:
        """聚合学术词汇统计"""
        if not vocabs:
            return {}

        aggregated = {
            'academic_connectors': sum(v['academic_connectors'] for v in vocabs) / len(vocabs),
            'academic_verbs': sum(v['academic_verbs'] for v in vocabs) / len(vocabs),
            'academic_adjectives': sum(v['academic_adjectives'] for v in vocabs) / len(vocabs)
        }

        return aggregated

    def _aggregate_structures(self, structures: List[Dict[str, bool]]) -> Dict[str, float]:
        """聚合结构信息(计算出现比例)"""
        if not structures:
            return {}

        aggregated = {}
        keys = structures[0].keys()

        for key in keys:
            aggregated[key] = sum(s.get(key, False) for s in structures) / len(structures)

        return aggregated

    def _format_aggregated_style_description(
        self,
        avg_length: float,
        length_dist: Dict[str, float],
        academic_vocab: Dict[str, float],
        passive_count: float,
        structure: Dict[str, float],
        sentence_count: float
    ) -> str:
        """
        格式化聚合后的风格描述

        Args:
            avg_length: 平均句长
            length_dist: 句长分布
            academic_vocab: 学术词汇统计
            passive_count: 被动语态数量
            structure: 结构信息
            sentence_count: 句子总数

        Returns:
            格式化的风格描述
        """
        description = f"""This academic article follows the style of Information Processing & Management journal, characterized by:

From the sentence perspective:
- The average sentence length is {avg_length:.1f} words, indicating {self._interpret_sentence_length(avg_length)}.
- Sentence length distribution shows {self._interpret_length_distribution(length_dist)}.
- The text employs approximately {passive_count:.0f} instances of passive voice per article on average, demonstrating {self._interpret_passive_voice(passive_count, sentence_count)}.

From the vocabulary perspective:
- The article uses {academic_vocab['academic_connectors']:.0f} academic connectors on average (e.g., however, moreover, furthermore, therefore, consequently, specifically), indicating {self._interpret_connectors(academic_vocab['academic_connectors'])}.
- Academic verbs appear {academic_vocab['academic_verbs']:.0f} times on average (e.g., demonstrate, indicate, reveal, propose, investigate, analyze, evaluate), showing {self._interpret_academic_verbs(academic_vocab['academic_verbs'])}.
- Academic adjectives occur {academic_vocab['academic_adjectives']:.0f} times on average (e.g., significant, comprehensive, robust, novel, effective, systematic), reflecting {self._interpret_academic_adjectives(academic_vocab['academic_adjectives'])}.

From the structural perspective:
- The article follows standard academic structure with {self._interpret_aggregated_structure(structure)}.
- The overall tone is objective, formal, and research-oriented, consistently avoiding colloquial expressions and maintaining scholarly rigor throughout.
- Citations and references are integrated systematically to support claims and arguments, following proper academic citation conventions.
- Technical terminology is used precisely and consistently, with clear definitions provided for domain-specific concepts.
- Arguments are evidence-based, supported by empirical data, theoretical frameworks, or prior research findings.
"""
        return description

    def _interpret_sentence_length(self, avg_length: float) -> str:
        """解释平均句长"""
        if avg_length < 15:
            return "concise and direct expression typical of technical writing"
        elif avg_length < 25:
            return "balanced complexity suitable for academic discourse"
        else:
            return "sophisticated sentence structures with detailed explanations"

    def _interpret_length_distribution(self, dist: Dict[str, float]) -> str:
        """解释句长分布"""
        if dist.get('long', 0) + dist.get('very_long', 0) > 0.5:
            return "a predominance of long and complex sentences characteristic of detailed academic analysis"
        elif dist.get('short', 0) > 0.5:
            return "predominantly short sentences for clarity and directness"
        else:
            return "a balanced mix of sentence lengths for varied rhythm and comprehensive coverage"

    def _interpret_passive_voice(self, passive_count: float, sentence_count: float) -> str:
        """解释被动语态使用"""
        if sentence_count == 0:
            return "minimal use of passive constructions"

        ratio = passive_count / sentence_count
        if ratio > 0.3:
            return "frequent use of passive voice, characteristic of objective academic reporting"
        elif ratio > 0.15:
            return "moderate use of passive voice to maintain scholarly objectivity"
        else:
            return "selective use of passive voice for emphasis on actions rather than actors"

    def _interpret_connectors(self, count: float) -> str:
        """解释连接词使用"""
        if count > 50:
            return "strong logical flow and coherent argumentation"
        elif count > 20:
            return "clear logical connections between ideas"
        else:
            return "basic logical structure"

    def _interpret_academic_verbs(self, count: float) -> str:
        """解释学术动词使用"""
        if count > 100:
            return "extensive use of formal academic discourse"
        elif count > 50:
            return "substantial academic vocabulary for rigorous analysis"
        else:
            return "moderate use of academic terminology"

    def _interpret_academic_adjectives(self, count: float) -> str:
        """解释学术形容词使用"""
        if count > 50:
            return "rich descriptive language emphasizing research significance"
        elif count > 25:
            return "appropriate use of evaluative language"
        else:
            return "restrained use of qualitative descriptors"

    def _interpret_aggregated_structure(self, structure: Dict[str, float]) -> str:
        """解释聚合后的文章结构"""
        sections = []
        for key, ratio in structure.items():
            if ratio >= 0.7:  # 70%以上的文章都有这个部分
                section_name = key.replace('has_', '').replace('_', ' ').title()
                sections.append(section_name)

        if len(sections) >= 5:
            return f"complete standard sections appearing in most articles: {', '.join(sections)}"
        elif len(sections) >= 3:
            return f"essential sections commonly found: {', '.join(sections)}"
        else:
            return "varied structural organization across articles"


def main():
    """主函数:分析data目录下所有期刊文章并生成综合风格描述"""
    analyzer = JournalStyleAnalyzer()
    data_dir = "/root/datasets/text_style_transfer/ipm_abstracts"

    # 分析所有文章并生成综合风格描述
    try:
        style_desc = analyzer.analyze_all_texts(data_dir)
        print("\n" + "=" * 80)
        print("AGGREGATED STYLE DESCRIPTION")
        print("=" * 80)
        print(style_desc)

        # 保存到文件
        output_file = os.path.join(os.path.dirname(data_dir), "aggregated_style_description.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(style_desc)
        print(f"\nStyle description saved to: {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\nStyle analysis completed!")


if __name__ == "__main__":
    main()
