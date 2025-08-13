import re
import ast

def fix_quote_syntax(list_string):
    """
    修正包含引號語法錯誤的列表字符串
    
    Args:
        list_string (str): 可能包含語法錯誤的列表字符串
        
    Returns:
        list: 正確解析後的列表
    """
    try:
        # 先嘗試直接解析
        return ast.literal_eval(list_string)
    except (SyntaxError, ValueError):
        # 如果直接解析失敗，進行修正
        pass
    
    # 移除外層的方括號
    content = list_string.strip()
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]
    
    # 分割元素，但要考慮引號內的逗號
    elements = []
    current_element = ""
    in_quotes = False
    quote_char = None
    i = 0
    
    while i < len(content):
        char = content[i]
        
        # 處理引號
        if char in ['"', "'"]:
            if not in_quotes:
                # 開始一個引號區域
                in_quotes = True
                quote_char = char
                current_element += char
            elif char == quote_char:
                # 結束引號區域
                in_quotes = False
                quote_char = None
                current_element += char
            else:
                # 在引號內但不是配對的引號
                current_element += char
        elif char == ',' and not in_quotes:
            # 元素分隔符
            elements.append(current_element.strip())
            current_element = ""
        else:
            current_element += char
        
        i += 1
    
    # 添加最後一個元素
    if current_element.strip():
        elements.append(current_element.strip())
    
    # 清理和規範化每個元素
    result = []
    for element in elements:
        element = element.strip()
        
        # 移除外層引號並處理內容
        if (element.startswith('"') and element.endswith('"')) or \
           (element.startswith("'") and element.endswith("'")):
            # 移除外層引號
            content = element[1:-1]
            
            # 處理轉義字符
            content = content.replace("\\'", "'").replace('\\"', '"')
            
            result.append(content)
        else:
            # 沒有引號的元素，直接添加
            result.append(element)
    
    return result

def safe_fix_quote_syntax(list_string):
    """
    更安全的版本，使用正則表達式處理
    
    Args:
        list_string (str): 可能包含語法錯誤的列表字符串
        
    Returns:
        list: 正確解析後的列表
    """
    try:
        # 先嘗試直接解析
        return ast.literal_eval(list_string)
    except (SyntaxError, ValueError):
        pass
    
    # 使用正則表達式提取字符串內容
    # 匹配被引號包圍的內容
    pattern = r"""(?:['"])((?:[^'"]|(?<=\\)['"])*)(?:['"])"""
    
    # 移除方括號
    content = list_string.strip()
    if content.startswith('[') and content.endswith(']'):
        content = content[1:-1]
    
    # 找到所有匹配的字符串
    matches = re.findall(pattern, content)
    
    if matches:
        return matches
    
    # 如果正則表達式失敗，使用備用方法
    return fallback_parse(content)

def fallback_parse(content):
    """
    備用解析方法
    """
    # 簡單的分割方法
    elements = []
    parts = content.split(',')
    
    for part in parts:
        part = part.strip()
        # 移除引號
        if part.startswith("'") or part.startswith('"'):
            part = part[1:]
        if part.endswith("'") or part.endswith('"'):
            part = part[:-1]
        elements.append(part)
    
    return elements

def normalize_to_list(input_data):
    """
    統一將各種格式轉換為標準的Python列表
    
    支持的輸入格式：
    - "['wildlife photography', 'nature's spectacle']" (字符串格式)
    - '["wildlife photography", "nature\'s spectacle"]' (字符串格式)
    - ['wildlife photography', 'nature's spectacle'] (實際列表)
    - ["wildlife photography", "nature's spectacle"] (實際列表)
    
    Args:
        input_data: 各種格式的輸入
        
    Returns:
        list: 標準的Python列表，內容為字符串
    """
    # 如果已經是列表，直接返回（Python會自動處理引號顯示）
    if isinstance(input_data, list):
        return [str(item) for item in input_data]  # 確保所有元素都是字符串
    
    # 如果是字符串，需要解析
    if isinstance(input_data, str):
        return fix_quote_syntax(input_data)
    
    # 其他類型，嘗試轉換為列表
    try:
        return list(input_data)
    except:
        return [str(input_data)]  # 如果無法轉換，就把它當作單個元素