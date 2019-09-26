# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
from sqlalchemy.orm import sessionmaker
from domria_scraper.models import db_connect, create_table, ApartmentsModel


class DomriaScraperPipeline(object):
    def __init__(self):
        engine = db_connect()
        create_table(engine)
        self.session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        session = self.session()
        apartmentsDB = ApartmentsModel(**item)

        try:
            session.add(apartmentsDB)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item
