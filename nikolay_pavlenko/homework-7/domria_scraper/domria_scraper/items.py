# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DomriaScraperItem(scrapy.Item):
    title = scrapy.Field()
    price_UAH = scrapy.Field()
    price_USD = scrapy.Field()
    description = scrapy.Field()
    photos_url = scrapy.Field()
    street_name = scrapy.Field()
    city_name = scrapy.Field()
    longitude = scrapy.Field()
    latitude = scrapy.Field()
    district_name = scrapy.Field()
    verified_apartment = scrapy.Field()
    rooms_count = scrapy.Field()
    floor = scrapy.Field()
    seller = scrapy.Field()
    floors_count = scrapy.Field()
    id = scrapy.Field()
    year = scrapy.Field()
    heating = scrapy.Field()
    square = scrapy.Field()
    district_name = scrapy.Field()
    wall_type = scrapy.Field()
    publishing_date = scrapy.Field()
    traffic_rating = scrapy.Field()
    infrastructure_rating = scrapy.Field()
    security_rating = scrapy.Field()
    ecology_rating = scrapy.Field()
    recreation_area_rating = scrapy.Field()
    build_character = scrapy.Field()
    rooms_character = scrapy.Field()
    communications = scrapy.Field()
    water = scrapy.Field()
    driveway = scrapy.Field()
    position_environment = scrapy.Field()
    building_number_str = scrapy.Field()
    type_of_selling = scrapy.Field()
