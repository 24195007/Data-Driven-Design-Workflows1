import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==== 配置部分 ====
# 1. 页面 URL
URL = "https://www.houzz.com/photos/home-design-ideas-phbr0-bp~"
# 2. 本地保存目录
DOWNLOAD_DIR = "houzz_images"


# ==== 初始化 webdriver ====
options = Options()
# options.add_argument("--headless")            # 无头模式
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

try:
    driver.get(URL)
    wait = WebDriverWait(driver, 10)
    # 等待主容器加载
    wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "div.browse-photo__result-set__container__grid")
    ))

    # ==== 向下滚动，直到页面底部，确保加载所有图片 ====
    scroll_pause = 2
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # ==== 获取所有图片卡片 ====
    cards = driver.find_elements(
        By.CSS_SELECTOR, "div.browse-photo__result-set__container__grid div.hz-photo-card.hz-track-me"
    )
    print(f"共找到 {len(cards)} 张图片卡片。")

    # ==== 确保本地目录存在 ====
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # ==== 逐张下载 ====
    for idx, card in enumerate(cards, start=1):
        try:
            img = card.find_element(By.CSS_SELECTOR, "img.hz-photo-card__img")
            # 优先使用高分辨率（srcset 中 2x 图），否则退回到 src
            srcset = img.get_attribute("srcset")
            if srcset and " 2x" in srcset:
                # 分割 srcset，取最后一个 URL
                img_url = srcset.split(",")[-1].strip().split(" ")[0]
            else:
                img_url = img.get_attribute("src")

            # 从 URL 中提取文件名
            filename = os.path.basename(img_url.split("?")[0])
            filepath = os.path.join(DOWNLOAD_DIR, filename)

            # 跳过已下载
            if os.path.exists(filepath):
                print(f"[{idx}/{len(cards)}] 已存在，跳过：{filename}")
                continue

            # 下载并写入文件
            resp = requests.get(img_url, timeout=10)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"[{idx}/{len(cards)}] 下载完成：{filename}")

        except Exception as e:
            print(f"[{idx}/{len(cards)}] 下载失败：{e}")

finally:
    driver.quit()
