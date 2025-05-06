import os
import random
import time
import re
import requests
from multiprocessing import Pool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from config import candidate_headers

# ========== 配置 ==========
SAVE_DIR = 'archdaily_images'
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_SCROLLS = 50  # 最多滚动次数
SCROLL_RETRY = 3  # 滚动到底检测重试次数
MAX_RETRIES = 5  # 下载图片最大重试次数
BACKOFF_FACTOR = 1.0  # 退避因子（秒），实际等待 = BACKOFF_FACTOR * 2^(attempt-1)
NUM_PROCESSES = 8  # 并发进程数


# ========== 辅助函数 ==========
def get_random_header():
    return random.choice(candidate_headers)


def scroll_to_bottom(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    return new_height > last_height


def extract_hd_image_urls(driver):
    imgs = driver.find_elements(By.CSS_SELECTOR, 'div.gridview__item img')
    urls = set()
    for img in imgs:
        srcset = img.get_attribute("srcset")
        if not srcset:
            continue
        matches = re.findall(r'(https://[^\s,]+)\s+2x', srcset)
        if matches:
            urls.add(matches[-1])
    return sorted(urls)


def download_image(args):
    url, idx = args
    ext = '.webp' if 'format=webp' in url else '.jpg'
    filename = os.path.join(SAVE_DIR, f"img_{idx:04d}{ext}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=get_random_header(), timeout=10)
            if resp.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(resp.content)
                print(f"✅ [OK] {filename} (尝试 {attempt})")
                return
            else:
                raise requests.RequestException(f"状态码 {resp.status_code}")
        except Exception as e:
            wait = BACKOFF_FACTOR * (2 ** (attempt - 1))
            print(f"⚠️ [重试 {attempt}/{MAX_RETRIES}] 下载失败：{url}\n    原因：{e}\n    {wait:.1f}s 后重试…")
            time.sleep(wait)
    print(f"❌ [失败] {url} 已超过 {MAX_RETRIES} 次重试，跳过。")


def crawl_images():
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # 可选：无界面
    chrome_options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=Service(), options=chrome_options)
    driver.get("https://www.archdaily.com/search/images")

    all_urls = set()
    scroll_times = 0
    while scroll_times < MAX_SCROLLS:
        new_urls = extract_hd_image_urls(driver)
        before = len(all_urls)
        all_urls.update(new_urls)
        after = len(all_urls)
        print(f"滚动第 {scroll_times + 1} 次，新增 {after - before} 张图，共计 {after} 张图")

        # 伪底检测：重试滚动
        success = False
        for retry in range(1, SCROLL_RETRY + 1):
            if scroll_to_bottom(driver):
                success = True
                break
            else:
                print(f"⚠️ 伪底检测，重试滚动 {retry}/{SCROLL_RETRY}…")
        if not success:
            print("📭 页面已真·到底")
            break

        scroll_times += 1

    driver.quit()
    return sorted(all_urls)


if __name__ == "__main__":
    urls = crawl_images()
    print(f"开始多进程下载 {len(urls)} 张图片…")
    args = [(url, i) for i, url in enumerate(urls)]
    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(download_image, args)
