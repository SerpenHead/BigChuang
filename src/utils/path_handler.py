import os

def handle_path(path):
    """根据操作系统返回对应平台风格的文件路径
    
    Args:
        path: 输入的文件路径字符串
        
    Returns:
        str: 转换后的平台相关路径
    """
    # 统一转为系统路径分隔符
    path = os.path.normpath(path)
    
    # Windows 下使用反斜杠,Linux/Mac 下使用正斜杠
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Linux/Mac
        return path.replace('\\', '/')
    
    
def contact_path(base_path, tail_path):
    """连接两个路径
    
    Args:
        base_path: 基础路径
        tail_path: 要连接的路径
        
    Returns:
        str: 连接后的完整路径
    """
    norm_base_path = os.path.normpath(base_path)
    norm_tail_path = os.path.normpath(tail_path)
    
    return os.path.join(norm_base_path, norm_tail_path)