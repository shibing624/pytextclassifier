# -*- coding: utf-8 -*-
"""
@description: 分类器
@author:XuMing
"""

import re

import jieba
from jieba import posseg


class DictClassifier:
    def __init__(self):
        self.__root_path = "data/dict/"
        jieba.load_userdict("data/dict/user.dict")  # 自定义分词词库

        # 情感词典
        self.__phrase_dict = self.__get_phrase_dict()
        self.__positive_dict = self.__get_dict(self.__root_path + "positive_dict.txt")
        self.__negative_dict = self.__get_dict(self.__root_path + "negative_dict.txt")
        self.__conjunction_dict = self.__get_dict(self.__root_path + "conjunction_dict.txt")
        self.__punctuation_dict = self.__get_dict(self.__root_path + "punctuation_dict.txt")
        self.__adverb_dict = self.__get_dict(self.__root_path + "adverb_dict.txt")
        self.__denial_dict = self.__get_dict(self.__root_path + "denial_dict.txt")

    def classify(self, sentence):
        return self.analyse_sentence(sentence)

    def analysis_file(self, file_path_in, file_path_out, encoding='utf-8', print_show=False, start=0, end=-1):
        results = []
        with open(file_path_in, 'r', encoding=encoding) as f:
            num_line = 0
            for line in f:
                # 语料开始位置
                num_line += 1
                if num_line < start:
                    continue

                results.append(self.analysis_sentence(line.strip(), file_path_out, print_show))

                # 语料结束位置
                if 0 < end <= num_line:
                    break

        return results

    def analyse_sentence(self, sentence, run_out_file_path=None, print_show=False):
        # 情感分析的数据结构
        comment_analysis = {"score": 0}

        # 评论分句
        clauses = self.__divide_sentence_to_clause(sentence + '%')

        # 对每个分句情感分析
        for i in range(len(clauses)):
            # 分析子句的数据结构
            sub_clause = self.__analyse_clause(clauses[i].replace("。", "."), run_out_file_path, print_show)
            # 将子句分析的数据结果添加到整体数据结构中
            comment_analysis["su-clause" + str(i)] = sub_clause
            comment_analysis["score"] += sub_clause["score"]

        if run_out_file_path is not None:
            # 将整句写到输出文件
            self.__write_out_file(run_out_file_path, "\n" + sentence + "\n")
            self.__output_analysis(comment_analysis, run_out_file_path)
            self.__write_out_file(run_out_file_path, str(comment_analysis) + "\n\n\n\n")

        if print_show:
            print("\n" + sentence)
            self.__output_analysis(comment_analysis)
            print(comment_analysis, end="\n\n\n")
        if comment_analysis["score"] > 0:
            return 1
        else:
            return 0

    def __divide_sentence_to_clause(self, sentence):
        clauses = self.__split_sentence(sentence)
        clauses[-1] = clauses[-1][:-1]
        return clauses

    def __analyse_clause(self, clauses, run_out_file_path, print_show):
        sub_clause = {"score": 0, "positive": [], "negative": [], "conjunction": [], "punctuation": [], "pattern": []}
        seg_result = posseg.lcut(clauses)

        # 输出分词结果
        if run_out_file_path is not None:
            self.__write_out_file(run_out_file_path, clauses + "\n")
            self.__write_out_file(run_out_file_path, str(seg_result) + "\n")
        if print_show:
            print(clauses)
            print(seg_result)

        # 判断句子：如果。。。就好了
        judgement = self.__is_clause_pattern_if_good(clauses)
        if judgement != "":
            sub_clause["pattern"].append(judgement)
            sub_clause["score"] -= judgement["value"]
            return sub_clause

        # 判断句子：是。。。不是。。。
        judgement = self.__is_clause_pattern_is_not(clauses)
        if judgement != "":
            sub_clause["pattern"].append(clauses)
            sub_clause["score"] -= judgement["value"]

        # 判断句子：短语
        judgement = self.__is_clause_pattern_phrase(clauses, seg_result)
        if judgement != "":
            sub_clause["score"] += judgement["score"]
            if judgement["score"] >= 0:
                sub_clause["positive"].append(judgement)
            elif judgement["score"] < 0:
                sub_clause["negative"].append(judgement)
            match_result = judgement["key"].split(":")[-1]
            i = 0
            while i < len(seg_result):
                if seg_result[i].word in match_result:
                    if i + 1 == len(seg_result) or seg_result[i + 1].word in match_result:
                        del (seg_result[i])
                        continue
                i += 1
        # 逐个分词
        for i in range(len(seg_result)):
            mark, result = self.__analyse_word(seg_result[i].word, seg_result, i)
            if mark == 0:
                continue
            elif mark == 1:
                sub_clause["conjunction"].append(result)
            elif mark == 2:
                sub_clause["punctuation"].append(result)
            elif mark == 3:
                sub_clause["positive"].append(result)
                sub_clause["score"] += result["score"]
            elif mark == 4:
                sub_clause["negative"].append(result)
                sub_clause["score"] -= result["score"]

        # 综合连词的情感值
        for conj in sub_clause["conjunction"]:
            sub_clause["score"] *= conj["value"]

        # 综合标点符号的情感值
        for punc in sub_clause["punctuation"]:
            sub_clause["score"] *= punc["value"]
        return sub_clause

    @staticmethod
    def __is_clause_pattern_if_good(clauses):
        re_pattern = re.compile(r".*(要|选)的.+(送|给).*")
        match = re_pattern.match(clauses)
        if match is not None:
            pattern = {"key": "要的是...给的是...", "value": 1}
            return pattern
        return ""

    @staticmethod
    def __is_clause_pattern_is_not(clauses):
        re_pattern = re.compile(r".*(如果|要是|希望).+就[\u4e00-\u9fa5]+(好|完美)了")
        match = re_pattern.match(clauses)
        if match is not None:
            pattern = {"key": "如果...就好了", "value": 1.0}
            return pattern
        return ""

    def __is_clause_pattern_phrase(self, clauses, seg_result):
        for phrase in self.__phrase_dict:
            keys = phrase.keys()
            to_compile = phrase["key"].replace("……", "[\u4e00-\u9fa5]*")

            if "start" in keys:
                to_compile = to_compile.replace("*", "{" + phrase["start"] + "," + phrase["end"] + "}")
            if "head" in keys:
                to_compile = phrase["head"] + to_compile
            match = re.compile(to_compile).search(clauses)
            if match is not None:
                is_continue = True
                pos = [flag for word, flag in posseg.cut(match.group())]
                if "between_tag" in keys:
                    if phrase["between_tag"] not in pos and len(pos) > 2:
                        is_continue = False

                if is_continue:
                    for i in range(len(seg_result)):
                        if seg_result[i].word in match.group():
                            try:
                                if seg_result[i + 1].word in match.group():
                                    return self.__emotional_word_analysis(
                                        phrase["key"] + ":" + match.group(), phrase["value"],
                                        [x for x, y in seg_result], i)
                            except IndexError:
                                return self.__emotional_word_analysis(
                                    phrase["key"] + ":" + match.group(), phrase["value"],
                                    [x for x, y in seg_result], i)
        return ""

    def __emotional_word_analysis(self, core_word, value, segments, index):
        # 情感词典内，则构建一个以情感词为中心的字典数据结构
        orientation = {"key": core_word, "adverb": [], "denial": [], "value": value}
        orientation_score = orientation["value"]

        # 判断三个前视窗内是否有否定词、副词
        view_window = index - 1
        if view_window > -1:
            # 前词是否是情感词
            if segments[view_window] in self.__negative_dict or segments[view_window] in self.__positive_dict:
                orientation["score"] = orientation_score
                return orientation

            # 前词是否是副词
            if segments[view_window] in self.__adverb_dict:
                adverb = {"key": segments[view_window], "position": 1,
                          "value": self.__adverb_dict[segments[view_window]]}
                orientation["adverb"].append(adverb)
                orientation_score *= self.__adverb_dict[segments[view_window]]
            # 前词是否是否定词
            elif segments[view_window] in self.__denial_dict:
                denial = {"key": segments[view_window], "position": 1,
                          "value": self.__denial_dict[segments[view_window]]}
                orientation["denial"].append(denial)
                orientation_score *= -1
        view_window = index - 2
        if view_window > -1:
            # 判断前一个词是否是情感词
            if segments[view_window] in self.__negative_dict or \
                            segments[view_window] in self.__positive_dict:
                orientation['score'] = orientation_score
                return orientation
            if segments[view_window] in self.__adverb_dict:
                adverb = {"key": segments[view_window], "position": 2,
                          "value": self.__adverb_dict[segments[view_window]]}
                orientation_score *= self.__adverb_dict[segments[view_window]]
                orientation["adverb"].insert(0, adverb)
            elif segments[view_window] in self.__denial_dict:
                denial = {"key": segments[view_window], "position": 2,
                          "value": self.__denial_dict[segments[view_window]]}
                orientation["denial"].insert(0, denial)
                orientation_score *= -1
                # 判断是否是“不是很好”的结构（区别于“很不好”）
                if len(orientation["adverb"]) > 0:
                    # 是，则引入调节阈值，0.3
                    orientation_score *= 0.3
        view_window = index - 3
        if view_window > -1:
            # 判断前一个词是否是情感词
            if segments[view_window] in self.__negative_dict or segments[view_window] in self.__positive_dict:
                orientation['score'] = orientation_score
                return orientation
            if segments[view_window] in self.__adverb_dict:
                adverb = {"key": segments[view_window], "position": 3,
                          "value": self.__adverb_dict[segments[view_window]]}
                orientation_score *= self.__adverb_dict[segments[view_window]]
                orientation["adverb"].insert(0, adverb)
            elif segments[view_window] in self.__denial_dict:
                denial = {"key": segments[view_window], "position": 3,
                          "value": self.__denial_dict[segments[view_window]]}
                orientation["denial"].insert(0, denial)
                orientation_score *= -1
                # 判断是否是“不是很好”的结构（区别于“很不好”）
                if len(orientation["adverb"]) > 0 and len(orientation["denial"]) == 0:
                    orientation_score *= 0.3
        # 添加情感分析值
        orientation['score'] = orientation_score
        # 返回的数据结构
        return orientation

    def __analyse_word(self, word, seg_result=None, index=-1):
        # 判断连词
        judgement = self.__is_word_conjunction(word)
        if judgement != "":
            return 1, judgement

        # 判断标点符号
        judgement = self.__is_word_punctuation(word)
        if judgement != "":
            return 2, judgement

        # 判断正向情感词
        judgement = self.__is_word_positive(word, seg_result, index)
        if judgement != "":
            return 3, judgement

        # 判断负向情感词
        judgement = self.__is_word_negative(word, seg_result, index)
        if judgement != "":
            return 4, judgement
        return 0, ""

    def __is_word_conjunction(self, word):
        if word in self.__conjunction_dict:
            conjunction = {"key": word, "value": self.__conjunction_dict[word]}
            return conjunction
        return ""

    def __is_word_punctuation(self, word):
        if word in self.__punctuation_dict:
            punctuation = {"key": word, "value": self.__punctuation_dict[word]}
            return punctuation
        return ""

    def __is_word_positive(self, word, seg_result, index):
        """
        判断分词在正向情感词典内
        :param word: 
        :param seg_result: 
        :param index: 
        :return: 
        """
        if word in self.__positive_dict:
            return self.__emotional_word_analysis(word, self.__positive_dict[word],
                                                  [x for x, y in seg_result], index)
        return ""

    def __is_word_negative(self, word, seg_result, index):
        """
        判断分词在负向情感词典内
        :param word: 
        :param seg_result: 
        :param index: 
        :return: 
        """
        if word in self.__negative_dict:
            return self.__emotional_word_analysis(word, self.__negative_dict[word],
                                                  [x for x, y in seg_result], index)
        return ""

    def __output_analysis(self, comment_analysis, run_out_file_path=None):
        output = "Score:" + str(comment_analysis["score"]) + "\n"
        for i in range(len(comment_analysis) - 1):
            output += "Sub-clause" + str(i) + ": "
            clause = comment_analysis["su-clause" + str(i)]
            if len(clause["conjunction"]) > 0:
                output += "conjunction:"
                for punctuation in clause["conjunction"]:
                    output += punctuation["key"] + " "
            if len(clause["positive"]) > 0:
                output += "positive:"
                for positive in clause["positive"]:
                    if len(positive["denial"]) > 0:
                        for denial in positive["denial"]:
                            output += denial["key"] + str(denial["position"]) + "-"
                    if len(positive['adverb']) > 0:
                        for adverb in positive["adverb"]:
                            output += adverb["key"] + str(adverb["position"]) + "-"
                    output += positive["key"] + " "
            if len(clause["negative"]) > 0:
                output += "negative:"
                for negative in clause["negative"]:
                    if len(negative["denial"]) > 0:
                        for denial in negative["denial"]:
                            output += denial["key"] + str(denial["position"]) + "-"
                    if len(negative["adverb"]) > 0:
                        for adverb in negative["adverb"]:
                            output += adverb["key"] + str(adverb["position"]) + "-"
                    output += negative["key"] + " "
            if len(clause["punctuation"]) > 0:
                output += "punctuation:"
                for pattern in clause["pattern"]:
                    output += pattern["key"] + " "
            output += "\n"
        if run_out_file_path is not None:
            self.__write_out_file(run_out_file_path, output)
        else:
            print(output)

    @staticmethod
    def __write_out_file(path, info, encoding="utf-8"):
        with open(path, "a", encoding=encoding) as f:
            f.write("%s" % info)

    @staticmethod
    def __split_sentence(sentence):
        pattern = re.compile("[，,。.%、！!？?;；～~…….… ]+")
        split_clauses = pattern.split(sentence.strip())
        punctuations = pattern.findall(sentence.strip())
        try:
            split_clauses.remove("")
        except ValueError:
            pass
        punctuations.append("")

        clauses = ["".join(x) for x in zip(split_clauses, punctuations)]
        return clauses

    def __get_phrase_dict(self):
        """
        取短语词典
        :return: 
        """
        sentiment_dict = []
        pattern = re.compile(r"\s+")
        with open(self.__root_path + "phrase_dict.txt", "r", encoding="utf-8") as f:
            for line in f:
                phrase = {}
                result = pattern.split(line.strip())
                if len(result) >= 2:
                    phrase["key"] = result[0]
                    phrase["value"] = float(result[1])
                    for i, temp_split in enumerate(result):
                        if i < 2:
                            continue
                        else:
                            a, b = temp_split.split(":")
                            phrase[a] = b
                    sentiment_dict.append(phrase)

        return sentiment_dict

    @staticmethod
    def __get_dict(path, encoding="utf-8"):
        """
        构建情感词典
        :param path: 
        :param encoding: 
        :return: 
        """
        sentiment_dict = {}
        pattern = re.compile(r"\s+")
        with open(path, encoding=encoding) as f:
            for line in f:
                result = pattern.split(line.strip())
                if len(result) == 2:
                    sentiment_dict[result[0]] = float(result[1])
        return sentiment_dict
