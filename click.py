import cv2
import numpy as np
import pyautogui
import time
import logging
import sys
import os
from datetime import datetime
from typing import Optional, Tuple, List
import json
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('button_monitor.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ButtonConfig:
    """按钮配置"""
    name: str
    image_path: str
    confidence: float = 0.85
    click_offset_x: int = 0
    click_offset_y: int = 0
    check_interval: float = 5.0
    enabled: bool = True

@dataclass
class ScrollConfig:
    """滚动配置（未找到按钮时执行一次）"""
    enabled: bool = False          # 是否启用自动滚动
    amount: int = -10               # 滚动量（负值向下，正值向上）

class ButtonMonitor:
    """按钮监视器 - 精确像素匹配版本，支持未找到时滚动"""
    
    def __init__(self, config_file: str = "button_config.json"):
        """
        初始化按钮监视器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.buttons = []
        self.running = False
        self.last_check_time = 0
        self.stats = {
            "total_checks": 0,
            "found_count": 0,
            "click_count": 0
        }
        
        # 滚动配置
        self.scroll_config = ScrollConfig()
        
        # 加载配置
        self.load_config()
        
        # 安全设置
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # 每次PyAutoGUI函数调用后暂停0.1秒
        
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 读取滚动配置
                scroll_data = config_data.get("scroll", {})
                self.scroll_config = ScrollConfig(
                    enabled=scroll_data.get("enabled", False),
                    amount=scroll_data.get("amount", -10)
                )
                
                # 读取按钮配置
                for btn_data in config_data.get("buttons", []):
                    btn = ButtonConfig(**btn_data)
                    
                    # 检查图片是否存在
                    if not os.path.exists(btn.image_path):
                        logger.error(f"按钮图片不存在: {btn.image_path}")
                        continue
                    
                    # 预加载按钮图片
                    btn_image = cv2.imread(btn.image_path, cv2.IMREAD_UNCHANGED)
                    if btn_image is None:
                        logger.error(f"无法加载按钮图片: {btn.image_path}")
                        continue
                    
                    # 添加到按钮列表
                    self.buttons.append({
                        "config": btn,
                        "image": btn_image,
                        "size": btn_image.shape[:2]  # (height, width, channels)
                    })
                    
                    logger.info(f"加载按钮: {btn.name} ({btn.image_path})")
                    
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                self.create_default_config()
        else:
            logger.info("配置文件不存在，创建默认配置")
            self.create_default_config()
    
    def create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            "scroll": {
                "enabled": True,      # 默认开启自动滚动
                "amount": -10          # 每次滚动幅度（负值向下）
            },
            "buttons": [
                {
                    "name": "确定按钮",
                    "image_path": "button.png",
                    "confidence": 0.85,
                    "click_offset_x": 0,
                    "click_offset_y": 0,
                    "check_interval": 5.0,
                    "enabled": True
                }
            ]
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            logger.info(f"已创建默认配置文件: {self.config_file}")
            
            # 重新加载配置
            self.load_config()
            
        except Exception as e:
            logger.error(f"创建配置文件失败: {e}")
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        捕获屏幕
        
        Args:
            region: (x, y, width, height)，如果为None则捕获整个屏幕
            
        Returns:
            屏幕截图
        """
        try:
            if region:
                x, y, width, height = region
                screenshot = pyautogui.screenshot(region=(x, y, width, height))
            else:
                screenshot = pyautogui.screenshot()
            
            # 转换为OpenCV格式 (BGR)
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # 如果有透明度通道，转换为BGR
            if screenshot_cv.shape[2] == 4:
                screenshot_cv = cv2.cvtColor(screenshot_cv, cv2.COLOR_BGRA2BGR)
            
            return screenshot_cv
            
        except Exception as e:
            logger.error(f"截图失败: {e}")
            raise
    
    def exact_match(self, screen: np.ndarray, template: np.ndarray) -> Tuple[bool, float, Tuple[int, int]]:
        """
        精确匹配 - 只做像素级别的严格匹配
        
        Args:
            screen: 屏幕截图
            template: 按钮模板
            
        Returns:
            (是否匹配, 匹配度, 匹配位置)
        """
        try:
            # 确保模板不大于屏幕
            screen_height, screen_width = screen.shape[:2]
            template_height, template_width = template.shape[:2]
            
            if template_height > screen_height or template_width > screen_width:
                logger.warning(f"模板({template_width}x{template_height})大于屏幕({screen_width}x{screen_height})")
                return False, 0.0, (0, 0)
            
            # 使用多种匹配方法，选择最好的结果
            methods = [
                (cv2.TM_CCOEFF_NORMED, "TM_CCOEFF_NORMED"),
                (cv2.TM_CCORR_NORMED, "TM_CCORR_NORMED"),
            ]
            
            best_confidence = 0.0
            best_location = (0, 0)
            best_method = ""
            
            for method, method_name in methods:
                # 执行模板匹配
                result = cv2.matchTemplate(screen, template, method)
                
                # 找到最佳匹配
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # 根据方法选择最佳匹配
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    confidence = 1 - min_val  # 对于平方差方法，值越小越好
                    location = min_loc
                else:
                    confidence = max_val
                    location = max_loc
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_location = location
                    best_method = method_name
            
            logger.debug(f"匹配结果: 方法={best_method}, 置信度={best_confidence:.3f}")
            
            return True, best_confidence, best_location
            
        except Exception as e:
            logger.error(f"匹配失败: {e}")
            return False, 0.0, (0, 0)
    
    def find_button(self, screen: np.ndarray, button_info: dict) -> Optional[Tuple[int, int]]:
        """
        在屏幕中查找按钮
        
        Returns:
            按钮位置 (x, y) 或 None
        """
        config = button_info["config"]
        template = button_info["image"]
        
        # 检查按钮是否启用
        if not config.enabled:
            return None
        
        # 执行匹配
        success, confidence, location = self.exact_match(screen, template)
        
        if success and confidence >= config.confidence:
            logger.info(f"找到按钮 [{config.name}]: 置信度={confidence:.3f}, 位置={location}")
            return location
        
        return None
    
    def click_button(self, button_info: dict, location: Tuple[int, int]) -> bool:
        """
        点击按钮
        
        Args:
            button_info: 按钮信息
            location: 按钮位置
            
        Returns:
            是否成功
        """
        try:
            config = button_info["config"]
            template = button_info["image"]
            
            # 计算点击位置（按钮中心 + 偏移）
            template_height, template_width = template.shape[:2]
            center_x = location[0] + template_width // 2 + config.click_offset_x
            center_y = location[1] + template_height // 2 + config.click_offset_y
            
            logger.info(f"准备点击 [{config.name}]: ({center_x}, {center_y})")
            
            # 缓慢移动鼠标（看起来更自然）
            pyautogui.moveTo(center_x, center_y, duration=0.3)
            time.sleep(0.1)
            
            # 点击按钮
            pyautogui.click()
            time.sleep(0.1)
            
            # 记录统计
            self.stats["click_count"] += 1
            
            logger.info(f"已点击按钮 [{config.name}]")
            return True
            
        except Exception as e:
            logger.error(f"点击失败: {e}")
            return False
    
    def check_once(self) -> bool:
        """
        执行一次检查
        
        Returns:
            是否找到并点击了任何按钮
        """
        self.stats["total_checks"] += 1
        
        try:
            # 捕获屏幕
            logger.debug("正在捕获屏幕...")
            screen = self.capture_screen()
            
            found_any = False
            
            # 检查每个按钮
            for button_info in self.buttons:
                config = button_info["config"]
                
                # 检查是否需要检查
                current_time = time.time()
                if hasattr(button_info, "last_check_time"):
                    time_since_last = current_time - button_info.last_check_time
                    if time_since_last < config.check_interval:
                        continue
                
                logger.debug(f"检查按钮: {config.name}")
                
                # 查找按钮
                location = self.find_button(screen, button_info)
                
                if location:
                    self.stats["found_count"] += 1
                    
                    # 点击按钮
                    success = self.click_button(button_info, location)
                    
                    if success:
                        found_any = True
                        
                        # 保存调试截图
                        self.save_debug_screenshot(screen, button_info, location)
                        
                        # 更新最后检查时间
                        button_info["last_check_time"] = current_time
                        
                        # 短暂的延迟，避免连续点击
                        time.sleep(0.5)
            
            return found_any
            
        except Exception as e:
            logger.error(f"检查失败: {e}")
            return False
    
    def save_debug_screenshot(self, screen: np.ndarray, button_info: dict, location: Tuple[int, int]):
        """保存调试截图"""
        try:
            config = button_info["config"]
            template = button_info["image"]
            
            # 创建调试目录
            debug_dir = "debug_screenshots"
            os.makedirs(debug_dir, exist_ok=True)
            
            # 在截图上标记按钮位置
            screen_marked = screen.copy()
            x, y = location
            h, w = template.shape[:2]
            
            # 绘制矩形框
            cv2.rectangle(screen_marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(screen_marked, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 添加文字
            text = f"{config.name}: {time.strftime('%H:%M:%S')}"
            cv2.putText(screen_marked, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 保存图片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{debug_dir}/{config.name}_{timestamp}.png"
            cv2.imwrite(filename, screen_marked)
            
            logger.debug(f"已保存调试截图: {filename}")
            
        except Exception as e:
            logger.warning(f"保存调试截图失败: {e}")
    
    def run(self):
        """运行监视器"""
        logger.info("=" * 50)
        logger.info("按钮监视器启动")
        logger.info(f"监视按钮数量: {len(self.buttons)}")
        if self.scroll_config.enabled:
            logger.info(f"未找到按钮时将自动滚动: 每次{self.scroll_config.amount}单位")
        else:
            logger.info("自动滚动已禁用")
        logger.info("按 Ctrl+C 停止程序")
        logger.info("=" * 50)
        
        self.running = True
        
        try:
            while self.running:
                try:
                    # 执行检查
                    found = self.check_once()
                    
                    if found:
                        # 如果找到了按钮，等待稍长时间（让按钮操作后的界面稳定）
                        wait_time = 2.0
                        logger.info(f"找到按钮，等待 {wait_time} 秒")
                        time.sleep(wait_time)
                    else:
                        # 未找到按钮，如果需要滚动则执行一次滚动
                        if self.scroll_config.enabled:
                            logger.info(f"未找到按钮，执行滚动: {self.scroll_config.amount}")
                            pyautogui.scroll(self.scroll_config.amount)
                            # 滚动后等待界面稳定
                            time.sleep(0.5)
                        else:
                            # 不滚动则等待较短时间
                            time.sleep(1.0)
                        
                    # 每10次检查打印一次统计信息
                    if self.stats["total_checks"] % 10 == 0:
                        self.print_stats()
                        
                except KeyboardInterrupt:
                    logger.info("接收到中断信号")
                    break
                except Exception as e:
                    logger.error(f"运行出错: {e}")
                    time.sleep(1)  # 出错后等待1秒再继续
                    
        finally:
            self.stop()
    
    def print_stats(self):
        """打印统计信息"""
        logger.info("=" * 30)
        logger.info("统计信息:")
        logger.info(f"总检查次数: {self.stats['total_checks']}")
        logger.info(f"找到按钮次数: {self.stats['found_count']}")
        logger.info(f"点击按钮次数: {self.stats['click_count']}")
        
        if self.stats['total_checks'] > 0:
            success_rate = (self.stats['found_count'] / self.stats['total_checks']) * 100
            logger.info(f"识别成功率: {success_rate:.1f}%")
        
        logger.info("=" * 30)
    
    def stop(self):
        """停止监视器"""
        self.running = False
        self.print_stats()
        logger.info("按钮监视器已停止")

def create_template_from_screen():
    """从屏幕创建模板图片的辅助函数"""
    print("=" * 50)
    print("创建模板图片")
    print("1. 将你要监视的按钮显示在屏幕上")
    print("2. 程序会在5秒后截图")
    print("3. 截图后，用鼠标拖动选择按钮区域")
    print("=" * 50)
    
    input("按回车键开始...")
    
    time.sleep(5)
    
    print("正在截图...")
    screenshot = pyautogui.screenshot()
    screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # 显示截图供用户选择
    cv2.imshow("选择按钮区域", screenshot_cv)
    
    # 设置鼠标回调
    selection = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection.clear()
            selection.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            selection.append((x, y))
            cv2.rectangle(screenshot_cv, selection[0], selection[1], (0, 255, 0), 2)
            cv2.imshow("选择按钮区域", screenshot_cv)
    
    cv2.setMouseCallback("选择按钮区域", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(selection) == 2:
        x1, y1 = selection[0]
        x2, y2 = selection[1]
        
        # 确保坐标正确
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # 提取按钮区域
        button_region = screenshot_cv[y1:y2, x1:x2]
        
        # 保存按钮图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"button_template_{timestamp}.png"
        cv2.imwrite(filename, button_region)
        
        print(f"按钮模板已保存: {filename}")
        print(f"尺寸: {button_region.shape[1]}x{button_region.shape[0]} 像素")
        
        return filename
    
    return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='精确按钮监视器（未找到时滚动）')
    parser.add_argument('--create', action='store_true', help='创建新的按钮模板')
    parser.add_argument('--config', default='button_config.json', help='配置文件路径')
    parser.add_argument('--test', help='测试单个按钮图片')
    
    args = parser.parse_args()
    
    # 创建模板模式
    if args.create:
        create_template_from_screen()
        return
    
    # 测试模式
    if args.test:
        print(f"测试按钮图片: {args.test}")
        
        if not os.path.exists(args.test):
            print(f"文件不存在: {args.test}")
            return
        
        # 加载按钮图片
        template = cv2.imread(args.test)
        if template is None:
            print(f"无法加载图片: {args.test}")
            return
        
        print(f"按钮尺寸: {template.shape[1]}x{template.shape[0]}")
        print("开始测试匹配...")
        
        monitor = ButtonMonitor(args.config)
        
        try:
            while True:
                print("\n" + "="*30)
                print("按 Ctrl+C 停止测试")
                input("按回车键开始测试匹配...")
                
                screen = monitor.capture_screen()
                
                # 创建测试按钮信息
                test_button = {
                    "config": ButtonConfig(
                        name="测试按钮",
                        image_path=args.test,
                        confidence=0.85
                    ),
                    "image": template,
                    "size": template.shape[:2]
                }
                
                location = monitor.find_button(screen, test_button)
                
                if location:
                    print(f"✓ 找到按钮! 位置: {location}")
                    
                    # 询问是否点击
                    answer = input("是否点击按钮? (y/n): ")
                    if answer.lower() == 'y':
                        monitor.click_button(test_button, location)
                else:
                    print("✗ 未找到按钮")
                    
        except KeyboardInterrupt:
            print("\n测试结束")
        return
    
    # 正常运行模式
    monitor = ButtonMonitor(args.config)
    
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()
