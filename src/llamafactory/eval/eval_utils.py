import re
import json
from typing import Optional, List
from .olympic_auto_scoring_judge import AutoScoringJudge
from functools import partial

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ') if len(completion.split('The answer is: '))>1 else completion.split('The answer is ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        print(f"extract_ans: {extract_ans}")
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    # except AssertionError:
    except Exception:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        # pdb.set_trace()
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2
    
def extract_gsm8k_answer_number(completion, flag=0):
    # 1. 按照空格和换行符分割字符串
    tokens = re.split(r'[\s\n\n]+', completion)
    # 2. 过滤掉不包含数字的单词(支持负数)
    if flag == 0:
        tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token) and not re.search(r'\d-[a-zA-Z]', token)]
    elif flag == 1:
        tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token) and not re.search(r'[a-zA-Z]', token)]
    else:
        tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token)]
    # tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token)]
    # 3. 去除与数字无关的符号（保留逗号和小数点）
    cleaned_numbers = [re.sub(r'[^\d,\.-]', '', token) for token in tokens_with_numbers]
    
    if cleaned_numbers:
        extracted_number = cleaned_numbers[-1].replace(',', '').strip('.')
        if extracted_number.count('.') > 1 or extracted_number.count('-') > 1 or not re.match(r'^-?\d*\.?\d*$', extracted_number):   # 小数点大于1，不符合规范
            return None
        try:
            return str(round(float(extracted_number))) 
        except:
            print(f"cannot convert to float: {extracted_number}")
            return None

    return  None


def extract_answer_between_boxed(completion, use_last_number=False):
    extract_ans = None
    extract_ans = remove_boxed(last_boxed_only_string(completion))  # 提取boxed里的内容
    if not extract_ans and use_last_number:   # use the last number
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, completion.replace(",", ""))
        if len(pred) >= 1:
            extract_ans = pred[-1]
    return extract_ans



def process_minerva_results(completion, answer, use_last_number=False, verbose=False):
    extract_ans = extract_answer_between_boxed(completion, use_last_number=use_last_number)
    # answer = answer.strip("<think>\n\n</think>").strip("\n")
    answer = answer.replace("<think>\n\n</think>", "").strip("\n")
    try:
        extract_ans = strip_string(extract_ans)
        answer = strip_string(answer)
        if "\\times" in extract_ans: # and re.match(r'e\d+', answer):
            extract_ans = normalize_scientific_str(extract_ans)
            answer = normalize_scientific_str(answer)

        # extract_ans = normalize_scientific_str()
        # answer = normalize_scientific_str()
        if verbose:
            print(extract_ans, answer)
        return extract_ans == answer
    except Exception:
        return is_equiv(extract_ans, answer)
import math
def normalize_scientific_str(s):
        # Step 1: 统一转换成 Python 可以识别的科学记数法
        s = re.sub(r'\s*\\times\s*10\^\{([^}]*)\}', lambda m: f'e{m.group(1)}', s)  # LaTeX \times 10^{...} -> e...
        s = re.sub(r'\s*x\s*10\^', 'e', s)  # x10^ -> e
        s = re.sub(r'([0-9])\s*x\s*', r'\1e', s)  # 1.1 x 10^... -> 1.1e...
        s = s.replace('^', '**')  # 替换幂符号

        # Step 2: 解析为浮点数
        try:
            value = float(eval(s))  # 将字符串表达式转为浮点数
        except Exception:
            raise ValueError(f"无法解析字符串: {s}")

        # Step 3: 转换为保留一位小数的科学记数法字符串
        if value == 0:
            return "0"
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = round(value / (10 ** exponent), 1)
        
        # 处理特殊情况：1.0 -> 不要 .0
        if mantissa == int(mantissa):
            mantissa = int(mantissa)

        return f"{mantissa}e{exponent}"

def process_math_aime_results(completion, answer, use_last_number=False, verbose=False):
    extract_ans = extract_answer_between_boxed(completion, use_last_number=use_last_number)
    # answer = answer.strip("<think>\n\n</think>").strip("\n")
    answer = answer.replace("<think>\n\n</think>", "").strip("\n")
    if is_equiv(extract_ans, answer, verbose=verbose):
        return True
    else:
        return False


def fraction_to_decimal(fraction_str):
    """将\\frac{x}{frac}转成相应的小数"""
    # 使用正则表达式匹配 \frac{分子}{分母} 格式
    match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', fraction_str)
    if not match:
        return None  # 如果格式不匹配，返回 None
    numerator = int(match.group(1))  # 提取分子
    denominator = int(match.group(2))  # 提取分母
    
    try:
        # 计算小数
        result = numerator / denominator
        return result
    except ZeroDivisionError:
        return None  # 分母为 0，返回 None

def strip_mawps_text(text: str):
    
    text_idx = text.find("text")   # 去掉单位，如\text{ gallons}
    text = text[:text_idx] if text_idx > 0 else text

    text = text.replace("pm", "")
    text = text.replace(",", "")
    text = text.replace("\!", "")
    text = text.replace("\\", "")
    text = text.replace("%", "")
    
    return text
    

def process_mawps_results(completion, answer, use_last_number=False, verbose=False):   
    """处理mawps数据集结果。注意数据集需要将提取到的答案转换成float"""
    extract_ans = extract_answer_between_boxed(completion, use_last_number=use_last_number)
    answer = answer.strip("<think>\n\n</think>").strip("\n")
    if "frac" in extract_ans:
        if not "frac" in answer:
            extract_ans = fraction_to_decimal(extract_ans)
            return True if extract_ans == float(answer) else False
        else:
            extract_ans = fraction_to_decimal(extract_ans)
            answer = fraction_to_decimal(answer)
            return True if extract_ans == float(answer) else False
    else:
        if "frac" in answer:
            answer = fraction_to_decimal(answer)
            return True if float(extract_ans) == answer else False
        try:
            extract_ans, answer = extract_ans.replace("$", ""), answer.replace("$", "")    # 去掉 $
            extract_ans = strip_mawps_text(extract_ans)
            if round(float(extract_ans), 2) == round(float(answer), 2):
                return True
            return False
        except:
            print(f"canot convert to float: {extract_ans}")
            return False


def process_gsm8k_results(completion, answer, use_last_number=False, verbose=False):
    """非推理模型+GSM8K"""
    extract_ans = extract_gsm8k_answer_number(completion, flag=0)
    if verbose:
            print(extract_ans, answer)
    if extract_ans:
        try:
            answer = answer.strip("<think>\n\n</think>").strip("\n").replace(",", "")
            if str(round(float(extract_ans.replace(",", "")))) == answer:
                return True
            else:
                return False
        except:
            return False
    else:
        return False

def process_gsm8k_results_v2(completion, answer, use_last_number=False, verbose=False):  # 用于推理模型GSM8K
    """推理模型+GSM8K"""
    extract_ans = extract_answer_between_boxed(completion, use_last_number=use_last_number)
    if extract_ans:
        extract_ans, answer = strip_string(extract_ans), strip_string(answer)
        if verbose:
            print(extract_ans, answer)
        try:
            if str(round(float(extract_ans.replace(",", "")))) == answer.replace(",", ""):
                return True
            else:
                # print(f"预测答案不正确。预测答案：{extract_ans}  标准答案：{answer}")
                return False
        except:
            return False
    else:
        return False


def is_braced(s: str, left="({", right=")}"):
    flag = True
    for l, r in zip(left, right):
        f = s.count(l) == s.count(r)  # and s.count(l) > 0
        flag = flag and f
    return flag

def split_equal(content: str):
    results = []
    if content.count("=") == 0:
        results.append(content.strip())

    elif content.count("=") == 1:
        # 检查等号左边的内容是否存在闭合（左右大/小括号都有），此时要进行分割
        idx = content.find("=")
        if is_braced(content[:idx]):  # 需要取等号右边的内容
            results.append(content.split("=")[-1].strip())
        else:
            results.append(content.strip())

    elif content.count("=") > 1:
        idx_list = [i for i, char in enumerate(content) if char == '=']
        if "," in content[idx_list[0]:idx_list[1]]:   # 类似 x=1, x=2，需要切分成1,2
            for i, idx in enumerate(idx_list):
                if i+1 < len(idx_list):
                    c = content[idx_list[i]+1:idx_list[i+1]]
                    results.append(c.split(",")[0])
                else:
                    results.append(content[idx+1:])
        else:  # 说明多个等号在字符串中不能分割，整个字符串是个等式
            results.append(content)
    
    return ', '.join(results)

def strip_text(text: str):
    text = text.lower().replace(" ", "")
    text = text.replace("dfrac", "frac")

    pattern = 'text\{(.*?)\}'
    if match := re.search(pattern, text):
        text = match.group(1)
    return text

def extract_college_math_ground_truth(answer: str):
    """"提取collegemath数据集样本的标准答案"""
    if "Therefore" in answer:  # $$\nX_{2}=\\left[\\begin{array}{l}\n0.367 \\\\\n0.4625 \\\\\n0.1705\n\\end{array}\\right]\n$$\n\nTherefore the probability of ending up in location 1 is 0.367.
        temp_ans = answer.split("Therefore")[-1]
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, temp_ans.replace(",", ""))  
        if len(pred) >= 1:
            extract_ans = pred[-1]  # 提取最后的数字
            return extract_ans
    
    # Step 1: 检查是否是奇数个 $，如果是，移除最左边的一个 $
    if answer.count('$') % 2 == 1:
        answer = answer.replace('$', '', 1)  # 只替换第一个 $
    # 正则匹配 $...$ 或 $$...$$ 中的内容（非贪婪）
    pattern = r'\$(\$?)(.*?)\$\1'
    # pattern = r'\$(.*?)\$|\$\$(.*?)\$\$'
    if match := re.search(pattern, answer, flags=re.DOTALL):
        content = match.group(2).strip()
        return split_equal(content)
    else:
        return answer


def process_college_math_results(completion, answer, use_last_number=False, verbose=False):
    extract_ans = extract_answer_between_boxed(completion)
    if extract_ans:
        extract_ans = split_equal(extract_ans)
    answer = extract_college_math_ground_truth(answer)
    
    if not extract_ans or not answer:
        return False
    
    extract_ans = strip_text(extract_ans)
    answer = strip_text(answer)
    if verbose:
        print(extract_ans, answer)

    return extract_ans == answer


def get_olympiad_completion_answer_list(pred_path: str, ori_path: str):
    with open(pred_path, "r") as f:
        lines = f.readlines()
        ori_data = [eval(line) for line in lines]
    
    with open(ori_path, "r") as f:
        pred_data = json.load(f)
    
    data = []
    for ori, pred in zip(ori_data, pred_data):
        data.append({**ori, **pred})
    
    completions = []
    answers = []
    for sample in data:
        answer_type = sample["answer_type"]
    
        if "Tuple" in answer_type:
            completions.append(sample["predict"])
            answers.append({"label": sample["final_answer"][0], "precision": 1e-8})
        else:
            if sample["error"]:
                if ',' in sample["error"]:
                    precisions = sample["error"].split(',')
                    precisions = [float(p) if p else 1e-8 for p in precisions]
                    completions.append(sample["predict"])
                    answers.append({"label": sample["final_answer"][0], "precision": precisions})
                else:
                    precision = float(sample["error"])
                    completions.append(sample["predict"])
                    answers.append({"label": sample["final_answer"][0], "precision": precision})
            else:
                completions.append(sample["predict"])
                answers.append({"label": sample["final_answer"][0], "precision": 1e-8})
    
    return completions, answers


def process_olympiad_results(completion: str, answer: dict, olympic_scorer, use_last_number=False, verbose=False):
    res = olympic_scorer.judge(completion, answer["label"], answer["precision"])
    return res


def process_math_vision_results(completion, answer, use_last_number=False, verbose=False): # math_vision 数据集，答案可能是A-E或数字字符串
    extract_ans = extract_answer_between_boxed(completion, use_last_number=use_last_number)
    if extract_ans:
        extract_ans = strip_string(extract_ans).upper()
        answer = strip_string(answer).upper()
        if verbose:
            print(extract_ans, answer)
        return extract_ans == answer
    else:
        return False

def process_batch(completion_batch: List[str], answer_batch: List[str], process_fn, verbose=False):
    true_or_false = []
    for completion, answer in zip(completion_batch, answer_batch):
        true_or_false.append(process_fn(completion, answer, use_last_number=True, verbose=verbose))
    return true_or_false

def auto_verify(completion_list: List[str], answer_list: List[str], dataset_type: str, verbose: bool = False):
    # 根据dataset_type确定process_fn
    if dataset_type == "gsm8k1":
        process_fn = process_gsm8k_results
    elif dataset_type == "gsm8k2":
        process_fn = process_gsm8k_results_v2
    elif dataset_type == "mawps":
        process_fn = process_mawps_results
    elif dataset_type == "college_math":
        process_fn = process_college_math_results
    elif dataset_type in ["math_500", "aime","math_500_0.20_0.3"]:
        process_fn = process_math_aime_results
    elif dataset_type == "minerva":
        process_fn = process_minerva_results
    elif dataset_type == "olympiad":
        olympic_scorer = AutoScoringJudge()
        process_fn = partial(process_olympiad_results, olympic_scorer=olympic_scorer)
    elif dataset_type == "math_vision":
        process_fn = process_math_vision_results
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")    
    
    return process_batch(completion_list, answer_list, process_fn, verbose)

if __name__ == "__main__":
    # predict = "To find out how many calls Tim deals with $10 during his 5-day work week, we need -50 to first calculate how many minutes he spends at work each day and then how many minutes he spends at work in a week.\n\n1. First, let's calculate how many minutes Tim spends at work each day. Since he spends 6 hours at work each day, we need to convert this to minutes. There are 60 minutes in an hour, so we multiply 6 hours by 60 minutes per hour.\n\n   6 hours * 60 minutes/hour = 360 minutes\n\n2. Now that we know Tim spends 360 minutes at work each day, we can calculate how many calls he deals with each day. We know it takes him 15 minutes to deal with a call, so we divide the total minutes he spends at work each day by the time it takes to deal with a call.\n\n   360 minutes / 15 minutes/call = 24 calls per day\n\n3. Finally, we need to calculate how many calls Tim deals with during his 5-day work week. We know he deals with 24 calls per day, so we multiply this by the number of days in his work week.\n\n   24 calls/day * 5 days/week = 120 calls per week\n\nTherefore, Tim deals with 120 calls during his 5-day work week."
    # predict = "Okay, so Josh is trying to flip a house, right? He buys it for $80,000 and then spends another $50,000 on repairs. After that, the value of the house increases by 150%. I need to figure out how much profit he made. Hmm, let me break this down step by step.\n\nFirst, let's figure out the total amount Josh invested in the house. He bought it for $80,000 and then spent $50,000 on repairs. So, adding those together, that's $80,000 + $50,000. Let me write that out:\n\nTotal Investment = Cost of House + Repair Costs\nTotal Investment = $80,000 + $50,000\nTotal Investment = $130,000\n\nOkay, so Josh spent a total of $130,000 on the house. Now, the value of the house increased by 150%. I need to calculate what the new value of the house is after this increase. \n\nWait, increasing by 150% means that the value goes up by 150% of the original price, right? So, the original price here is the price after repairs, which is $130,000. So, 150% of $130,000 is how much? Let me compute that.\n\nFirst, 100% of $130,000 is just $130,000. Then, 50% of $130,000 is half of that, which is $65,000. So, 150% would be 100% + 50%, which is $130,000 + $65,000 = $195,000.\n\nSo, the value of the house increased by $195,000. Therefore, the new value of the house is the original investment plus the increase. Let me write that:\n\nNew Value = Total Investment + Increase\nNew Value = $130,000 + $195,000\nNew Value = $325,000\n\nWait, hold on. Is that right? Because if the value increased by 150%, does that mean it's 150% more than the original, or 150% of the original? Hmm, I think it's 150% of the original. So, 150% of $130,000 is $195,000, so adding that to the original $130,000 gives $325,000. That seems correct.\n\nNow, to find the profit, I need to subtract the total investment from the new value. So, Profit = New Value - Total Investment.\n\nLet me compute that:\n\nProfit = $325,000 - $130,000\nProfit = $200,000\n\nSo, Josh made a profit of $200,000. Hmm, that seems like a lot, but flipping houses can be profitable if everything goes right.\n\nWait, let me double-check my calculations to make sure I didn't make a mistake. \n\nTotal Investment: $80,000 + $50,000 = $130,000. That seems right.\n\n150% increase on $130,000: 150% is 1.5 in decimal. So, 1.5 * 130,000 = 195,000. So, the increase is $195,000. Adding that to the original $130,000 gives $325,000. Then, subtracting the investment gives $200,000 profit. Yeah, that seems consistent.\n\nAlternatively, another way to think about it is that a 150% increase means the new value is 100% + 150% = 250% of the original investment. So, 250% of $130,000 is 2.5 * 130,000 = 325,000. Then, subtracting the original investment gives the profit. Yep, same result.\n\nSo, I think my calculations are correct. Josh made a $200,000 profit by flipping the house.\n\n**Final Answer**\nJosh made a profit of \\boxed{200000} dollars.\n</think>\n\nJosh buys a house for $80,000 and spends $50,000 on repairs. The total investment is calculated as follows:\n\n\\[\n\\text{Total Investment} = \\$80,000 + \\$50,000 = \\$130,000\n\\]\n\nThe value of the house increases by 150%. To find the new value, we calculate 150% of the total investment and add it to the original investment:\n\n\\[\n\\text{Increase} = 1.5 \\times \\$130,000 = \\$195,000\n\\]\n\\[\n\\text{New Value} = \\$130,000 + \\$195,000 = \\$325,000\n\\]\n\nThe profit is the difference between the new value and the total investment:\n\n\\[\n\\text{Profit} = \\$325,000 - \\$130,000 = \\$200,000\n\\]\n\nThus, Josh made a profit of \\boxed{200000} dollars."
    # predict = "Okay, so Annika went to the town fair with $50. Hmm, that's a good amount for a fair. She spent half of it on food and snacks. Let me figure out how much that is. Half of 50 is... let's see, 50 divided by 2 is 25. So she spent $25 on food and snacks. \n\nNow, after that, she also spent an additional $10 on rides. So I need to subtract that too. Let me add up the total she spent. She spent $25 on food and $10 on rides. That's 25 plus 10, which equals 35. So she spent a total of $35.\n\nWait, let me make sure I did that right. Half of 50 is 25, right? Yeah, because 25 times 2 is 50. Then she spent another 10, so 25 plus 10 is definitely 35. Okay, that seems correct.\n\nNow, to find out how much she has left, I need to subtract the total she spent from the amount she brought. She started with $50 and spent $35. So, 50 minus 35. Let me do that subtraction. 50 minus 30 is 20, and then minus 5 more is 15. So, 50 minus 35 is 15.\n\nWait, is that right? Let me check again. 35 plus 15 is 50, yes. So that makes sense. So she has $15 left after spending on food, snacks, and rides.\n\nI think that's it. She started with 50, spent 25 on food, then 10 on rides, totaling 35, so she has 15 left. Yeah, that seems correct.\n\n**Final Answer**\nThe amount left is \\boxed{15} dollars.\n</think>\n\nAnnika brought $50 to the town fair. She spent half of it on food and snacks, which is calculated as follows:\n\n\\[\n\\frac{50}{2} = 25\n\\]\n\nSo, she spent $25 on food and snacks. Additionally, she spent $10 on rides. The total amount she spent is:\n\n\\[\n25 + 10 = 35\n\\]\n\nTo find out how much she has left, we subtract the total amount spent from the initial amount:\n\n\\[\n50 - 35 = 15\n\\]\n\nThus, the amount left is \\boxed{15} dollars."
    # extracted_answer = extract_gsm8k_answer_number(predict)
    # answer = "$\\sum_{n=0}^{\\infty} \\frac{(-1)^{n}}{2 n+1} x^{2 n+1}$"
    # answer = "Solution is:\n\n$$\n(1-i) \\sqrt{2},-(1+i) \\sqrt{2},-(1-i) \\sqrt{2},(1+i) \\sqrt{2}\n$$"
    # answer = extract_college_math_ground_truth(answer)

    
    paths = [
        "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-hf/minerva_formatted/generated_predictions.jsonl"
    ]

    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-hf/minerva_formatted/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/math_500/SAFE_Entropy_loss_detach_v2/lam_0.5-lr_7.5e-6-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/naive_entropy/lr_7.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/math_500/dirichlet_entropy/lr_7.5e-6-all_tokens-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    # path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_eata/math_500/naive_entropy/lr_5e-6-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    # path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_ttl/math_500/lr_7.5e-6-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-vllm/math_500/math_500-template_llama3-temperature_0.0-max_new_tokens_8192-500samples-generations.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_adaptive_k/relative_entropy_as_thres-without_abs/aime24_formatted/SAFE_Entropy_loss_detach_v2/lr_1.25e-5-new_tokens_32-N0_8-K_max_24-K_min_2-initial_delta_0.02-patient_m_2-threshold_low_0.02-coef_0.3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/aime24_formatted/naive_entropy/lr_1.25e-5-new_tokens_32-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_tent/aime24_formatted/dirichlet_entropy/lr_1.25e-5-new_tokens_32-seed_42-v2/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_eata/aime24_formatted/naive_entropy/lr_1.25e-5-all_tokens-threshold_0.4-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/online_ttl/aime24_formatted/lr_1.25e-5-all_tokens-threshold_3-seed_42/predict-temperature_0.0-max_new_tokens_8192/generated_predictions.jsonl"
    path = "/hujinwu/wyf/projects/zhangzitian/projects/LLaMA-Factory-1cfe429/saves/AAAI26/Llama-3.1-8B-Instruct/base_model_results-vllm/aime24_formatted/aime24_formatted-template_llama3-temperature_0.0-max_new_tokens_8192-30samples-generations.jsonl"
    ress = []
    preds = []
    labels = []
    indices = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:]):
            data = eval(line)
            pred = data["predict"]
            label = data["label"]
            preds.append(pred)
            labels.append(label)
            res = process_math_aime_results(pred, label, use_last_number=True, verbose=True)
            # res = process_minerva_results(pred, label, use_last_number=True, verbose=True)
            if res:
                indices.append(i+1)
                ress.append(res)
            # ress.append(res)
    print(indices)    
    print(sum(ress)/len(lines))
