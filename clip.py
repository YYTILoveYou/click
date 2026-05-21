#!/usr/bin/env python3
"""
clipboard_comment_cleaner.py

监听剪贴板并移除 C/C++/Java 的 // 和 /* */ 注释，以及 Python 的 # 注释（和可选的三引号字符串）
"""
import sys
import time
try:
    from pygments import lex
    from pygments.token import Token
    from pygments.lexers import get_lexer_by_name, guess_lexer
    PYGMENTS_AVAILABLE = True
except Exception:
    PYGMENTS_AVAILABLE = False

def _init_clipboard_backends():
    try:
        import pyperclip
        def get_clipboard():
            return pyperclip.paste()
        def set_clipboard(text):
            pyperclip.copy(text)
        return get_clipboard, set_clipboard
    except Exception:
        try:
            import tkinter as tk
            def get_clipboard():
                root = tk.Tk()
                root.withdraw()
                try:
                    val = root.clipboard_get()
                finally:
                    root.destroy()
                return val
            def set_clipboard(text):
                root = tk.Tk()
                root.withdraw()
                root.clipboard_clear()
                root.clipboard_append(text)
                root.update()
                root.destroy()
            return get_clipboard, set_clipboard
        except Exception:
            def _no_clip_get():
                raise RuntimeError('需要安装 pyperclip 或在 GUI 环境下运行')
            def _no_clip_set(_):
                raise RuntimeError('需要安装 pyperclip 或在 GUI 环境下运行')
            return _no_clip_get, _no_clip_set

get_clipboard, set_clipboard = _init_clipboard_backends()

def remove_c_style_comments(code: str) -> str:
    res = []
    i = 0
    n = len(code)
    in_str = False
    str_quote = ''
    while i < n:
        ch = code[i]
        nxt = code[i+1] if i+1 < n else ''
        if not in_str:
            if ch == '/' and nxt == '/':
                i += 2
                while i < n and code[i] != '\n':
                    i += 1
                continue
            if ch == '/' and nxt == '*':
                i += 2
                while i < n-1 and not (code[i] == '*' and code[i+1] == '/'):
                    i += 1
                i += 2 if i < n-1 else 0
                continue
            if ch == '"' or ch == "'":
                in_str = True
                str_quote = ch
                res.append(ch)
                i += 1
                continue
            res.append(ch)
            i += 1
        else:
            res.append(ch)
            if ch == '\\':
                if i+1 < n:
                    res.append(code[i+1])
                    i += 2
                    continue
            if ch == str_quote:
                in_str = False
                str_quote = ''
            i += 1
    return ''.join(res)

def remove_python_comments(code: str, remove_triple: bool = True) -> str:
    res = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if ch == '#':
            while i < n and code[i] != '\n':
                i += 1
            continue
        if ch in ('"', "'"):
            if i+2 < n and code[i:i+3] == ch*3:
                if remove_triple:
                    delim = ch*3
                    j = i+3
                    while j < n-2 and code[j:j+3] != delim:
                        j += 1
                    i = j+3 if j < n-2 else n
                    continue
                else:
                    res.append(ch*3)
                    i += 3
                    while i < n-2 and code[i:i+3] != ch*3:
                        res.append(code[i])
                        i += 1
                    if i < n-2:
                        res.append(code[i:i+3])
                        i += 3
                    continue
            else:
                res.append(ch)
                i += 1
                while i < n:
                    res.append(code[i])
                    if code[i] == '\\':
                        i += 1
                        if i < n:
                            res.append(code[i])
                            i += 1
                        continue
                    if code[i] == ch:
                        i += 1
                        break
                    i += 1
                continue
        res.append(ch)
        i += 1
    return ''.join(res)


def remove_comments_pygments(text: str, language: str = 'auto', remove_triple: bool = True) -> str:
    """Use Pygments lexer to remove comment tokens and optional docstring tokens

    Falls back to preserving newlines inside removed tokens to keep layout
    reasonably stable
    """
    if not PYGMENTS_AVAILABLE:
        raise RuntimeError('pygments not available')

    # map our language keys to pygments lexer names when obvious
    lang_map = {
        'c': 'c',
        'cpp': 'cpp',
        'java': 'java',
        'javascript': 'javascript',
        'js': 'javascript',
        'python': 'python',
        'go': 'go',
    }
    lexer = None
    try:
        lexer_name = lang_map.get(language)
        if lexer_name:
            lexer = get_lexer_by_name(lexer_name)
        else:
            lexer = guess_lexer(text)
    except Exception:
        try:
            lexer = guess_lexer(text)
        except Exception:
            lexer = None

    if lexer is None:
        # no lexer -> fallback to simple removers
        if language == 'python':
            return remove_python_comments(text, remove_triple=remove_triple)
        return remove_c_style_comments(text)

    out = []
    for toktype, value in lex(text, lexer):
        # remove comment tokens but keep newline characters inside them
        try:
            if toktype in Token.Comment:
                nl = value.count('\n')
                if nl:
                    out.append('\n' * nl)
                continue
        except Exception:
            pass

        # optionally remove docstring-like tokens (python triple-quoted)
        try:
            if remove_triple and (toktype in Token.Literal.String.Doc or toktype in Token.String.Doc):
                nl = value.count('\n')
                if nl:
                    out.append('\n' * nl)
                continue
        except Exception:
            pass

        out.append(value)

    return ''.join(out)

def _match_any(text: str, patterns) -> bool:
    return any(pattern in text for pattern in patterns)

def detect_language(text: str) -> str:
    low = text.lower()
    # Go: package + func
    if 'package ' in low and 'func ' in low:
        return 'go'
    # Java heuristics
    if 'public static void main' in low or 'system.out.println' in low or 'import java.' in low:
        return 'java'
    # Python: prefer syntax that is uncommon in Java/C/JS
    python_markers = (
        'def ',
        'async def ',
        'from ',
        'print(',
        'self',
        '__init__',
        'elif ',
        'except ',
        'with ',
        'yield ',
        'lambda ',
    )
    if _match_any(low, python_markers) and '#include' not in text:
        return 'python'
    # C/C++ heuristics: explicit markers first, then common C syntax
    if _match_any(low, ('#include', '#define', 'printf(', 'scanf(', 'fprintf(', 'malloc(', 'free(', 'std::', 'cout', 'cin', 'namespace ', 'template<', 'typedef ', 'struct ', 'enum ', 'nullptr', 'using namespace ')):
        return 'c'
    # JavaScript heuristics: require unmistakable JS syntax, avoid const/let/var alone
    js_markers = (
        'console.',
        'document.',
        'window.',
        'module.exports',
        'exports.',
        'require(',
        'export default',
        'export {',
        '=>',
        'function ',
        'import ',
    )
    if _match_any(low, js_markers):
        if 'import ' in low and '#include' in text:
            return 'c'
        return 'javascript'
    return 'c'

def remove_comments(text: str, language: str = 'auto', remove_triple: bool = True) -> str:
    if language == 'auto':
        language = detect_language(text)
    # prefer a lexer-based removal when available for better accuracy
    if PYGMENTS_AVAILABLE:
        try:
            return remove_comments_pygments(text, language, remove_triple=remove_triple)
        except Exception:
            # fallback to simple removers on any lexer failure
            pass
    if language == 'python':
        return remove_python_comments(text, remove_triple=remove_triple)
    return remove_c_style_comments(text)

def watch_clipboard(interval: float = 0.8, remove_triple: bool = True):
    try:
        last = get_clipboard()
    except Exception as e:
        print('剪贴板访问失败:', e)
        last = ''
    print('开始持续监测剪贴板，按 Ctrl+C 停止')
    try:
        while True:
            try:
                cur = get_clipboard()
            except Exception:
                time.sleep(interval)
                continue
            if cur != last:
                cleaned = remove_comments(cur, 'auto', remove_triple)
                if cleaned != cur:
                    try:
                        set_clipboard(cleaned)
                        print('已清理注释，检测到语言：', detect_language(cur))
                    except Exception as e:
                        print('无法设置剪贴板:', e)
                last = cleaned
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\n已停止监听')


# simplified: run watch loop directly by default (no tray, no CLI args)

def run_self_test():
    cases = [
        ('c', 'int main(){ const int value = 1; return value; }'),
        ('java', 'public class Main { public static void main(String[] args) { System.out.println(1); } }'),
        ('python', 'def hello():\n    print("hi")\n'),
    ]
    results = []
    for expected, sample in cases:
        got = detect_language(sample)
        results.append((expected, got))
        if got != expected:
            raise AssertionError(f'{expected} -> {got}')
    return results

def main():
    # 直接进入持续监听模式（无命令行参数）
    try:
        watch_clipboard()
    except Exception as e:
        print('发生错误:', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
