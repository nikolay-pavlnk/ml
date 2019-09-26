# -*- coding: utf-8 -*-
import os

BOT_NAME = "domria_scraper"

DATABASE = {
    "drivername": "postgres",
    "host": "localhost",
    "port": "5432",
    "username": "apart_db",
    "password": "rbhgbljy1",
    "database": "apart_db",
}

SPIDER_MODULES = ["domria_scraper.spiders"]

ITEM_PIPELINES = {"domria_scraper.pipelines.DomriaScraperPipeline": 300}
