# -*- coding: utf-8 -*-
from scrapy import Request, Spider
from domria_scraper.items import DomriaScraperItem
from scrapy.http import HtmlResponse
from domria_scraper.utils import *
import requests
import json


class DomRiaSpider(Spider):
    name = "domria_spider"

    def __init__(self, start_page, max_page):
        allowed_domains = ["https://dom.ria.com/"]
        self.start_urls = [
            PAGE_TO_SCRAPY.format(i) for i in range(start_page, max_page)
        ]

    def start_requests(self):
        for url in self.start_urls:
            yield Request(url=url, callback=self.start_parse, dont_filter=True)

    def start_parse(self, response):

        for page in response.css("#catalogResults .realtyPhoto::attr(href)").getall():
            apartment_page = response.urljoin(page)
            yield Request(url=apartment_page, callback=self.parse, dont_filter=True)

    def parse(self, response):
        self.apartment = DomriaScraperItem()

        data = response.xpath(
            '//script[contains(., "window.__INITIAL_STATE")]/text()'
        ).extract_first()
        data = json.loads(filter_json(str(data)))
        realty = data["dataForFinalPage"]["realty"]
        self.apartment.update(
            find_features(
                data["dataForFinalPage"]["realty"]["secondaryParams"],
                "groupName",
                "items",
            )
        )
        self.apartment["title"] = data["dataForFinalPage"]["tagH"]

        keys = clean(response.css("#additionalInfo .mt-20 .label.grey::text").getall())
        values = clean(response.css("#additionalInfo .mt-20 .boxed::text").getall())
        self.apartment.update({feature: realty.get(feature) for feature in MAP_INFO})
        self.apartment.update(create_dict(keys, values, MAP_ADDITIONAL_INFO))
        self.apartment["publishing_date"] = realty["publishing_date"].split(" ")[0]

        price_1 = response.css(".price::text").get()
        price_2 = response.css(".ml-30 .grey.size13 span::text").get()
        prices = get_prices(price_1, price_2)

        self.apartment["price_UAH"] = prices[0]
        self.apartment["price_USD"] = prices[1]
        self.apartment["photos_url"] = response.css(
            ".tumbs.unstyle img::attr(src)"
        ).getall()

        self.apartment["id"] = response.css(
            ".greyLight.size13.unstyle .mt-15:nth-child(2) b::text"
        ).get()

        self.apartment["square"] = (
            response.css(".mt-15.boxed.v-top:nth-child(4) .indent::text")
            .get()
            .split(f"\n{20*' '}")[1]
            .split("\n")[0]
        )

        verified = response.css(".unstyle .ml-30 .blue::text").get() is not None

        self.apartment.update({"verified_apartment": verified})

        self.apartment.update(
            find_features(realty["mainCharacteristics"]["chars"], "name", "value")
        )
        self.apartment["type_of_selling"] = realty["mainCharacteristics"]["dashes"]

        new_response = requests.get(
            f'https://dom.ria.com/ru/rate-region-stat/{realty["district_id"]}/'
        )
        new_response = HtmlResponse(
            url="", body=new_response.content.decode("utf-8"), encoding="utf-8"
        )

        marks = new_response.css(
            ".main-rate.grid.boxed .orange.bold.size22::text"
        ).getall()
        categories = clean(
            new_response.css(".main-rate.grid.boxed .rows.list .indent::text").getall()
        )
        self.apartment.update(create_dict(categories, marks, MAP_RATING))

        yield self.apartment
