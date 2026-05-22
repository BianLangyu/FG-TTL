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
    tokens = re.split(r'[\s\n\n]+', completion)
    if flag == 0:
        tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token) and not re.search(r'\d-[a-zA-Z]', token)]
    elif flag == 1:
        tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token) and not re.search(r'[a-zA-Z]', token)]
    else:
        tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token)]
    # tokens_with_numbers = [token for token in tokens if re.search(r'-?\d', token)]
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
    extract_ans = remove_boxed(last_boxed_only_string(completion)) 
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
        s = re.sub(r'\s*\\times\s*10\^\{([^}]*)\}', lambda m: f'e{m.group(1)}', s)  # LaTeX \times 10^{...} -> e...
        s = re.sub(r'\s*x\s*10\^', 'e', s) 
        s = re.sub(r'([0-9])\s*x\s*', r'\1e', s)
        s = s.replace('^', '**') 


        try:
            value = float(eval(s)) 
        except Exception:
            raise ValueError(f"无法解析字符串: {s}")


        if value == 0:
            return "0"
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = round(value / (10 ** exponent), 1)

        if mantissa == int(mantissa):
            mantissa = int(mantissa)

        return f"{mantissa}e{exponent}"

def process_math_aime_results(completion, answer, use_last_number=False, verbose=False):
    extract_ans = extract_answer_between_boxed(completion, use_last_number=use_last_number)
    answer = answer.replace("<think>\n\n</think>", "").strip("\n")
    if is_equiv(extract_ans, answer, verbose=verbose):
        return True
    else:
        return False


def fraction_to_decimal(fraction_str):
    """将\\frac{x}{frac}转成相应的小数"""
    match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', fraction_str)
    if not match:
        return None
    numerator = int(match.group(1))
    denominator = int(match.group(2)) 
    
    try:
        result = numerator / denominator
        return result
    except ZeroDivisionError:
        return None  

def strip_mawps_text(text: str):
    
    text_idx = text.find("text")  
    text = text[:text_idx] if text_idx > 0 else text

    text = text.replace("pm", "")
    text = text.replace(",", "")
    text = text.replace("\!", "")
    text = text.replace("\\", "")
    text = text.replace("%", "")
    
    return text
    

def process_mawps_results(completion, answer, use_last_number=False, verbose=False):   
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
        idx = content.find("=")
        if is_braced(content[:idx]): 
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
        else: 
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
    if "Therefore" in answer:  
        temp_ans = answer.split("Therefore")[-1]
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, temp_ans.replace(",", ""))  
        if len(pred) >= 1:
            extract_ans = pred[-1]  
            return extract_ans
    
    if answer.count('$') % 2 == 1:
        answer = answer.replace('$', '', 1) 
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

