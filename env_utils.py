import os

from dotenv import load_dotenv

load_dotenv(override=True)

XIAOAI_API_KEY = os.getenv("XIAOAI_API_KEY")
XIAOAI_BASE_URL = os.getenv("XIAOAI_BASE_URL")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")