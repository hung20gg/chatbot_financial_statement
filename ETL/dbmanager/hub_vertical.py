# @pphanhh

from langchain_chroma import Chroma
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import re

from dotenv import load_dotenv
import os

load_dotenv()

import logging
import pandas as pd
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from ..connector import *
from .abstracthub import BaseDBHUB


class HubVerticalBase(BaseDBHUB):
    pass 



class HubVerticalUniversal(BaseDBHUB):
    pass