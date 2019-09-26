MAP_INFO = [
    "rooms_count",
    "district_name",
    "description",
    "longitude",
    "latitude",
    "wall_type",
    "district_name",
    "building_number_str",
    "floor",
    "city_name",
    "floors_count",
    "street_name",
]

MAP_RATING = {
    "Транспортное сообщение": "traffic_rating",
    "Социальная инфраструктура в районе": "infrastructure_rating",
    "Зоны для отдыха и занятий спортом": "recreation_area_rating",
    "Безопасность района": "security_rating",
    "Экологическая обстановка": "ecology_rating",
}

MAP_ADDITIONAL_INFO = {
    "seller": "Тип предложения",
    "heating": "Отопление",
    "driveway": "подъезд",
    "position_environment": "положение и окружение",
    "build_character": "характеристика здания",
    "rooms_character": "характеристика помещения",
    "communications": "коммуникации",
    "water": "вода",
    "year": "Год постройки",
}

PAGE_TO_SCRAPY = "https://dom.ria.com/prodazha-kvartir/?page={0}"


def get_prices(price_1: str, price_2: str):
    """
    Get price UAH, price USD
    """
    if price_1[-2] == "$":
        uah, usd = price_2[:-4], price_1[:-3]
    else:
        uah, usd = price_1[:-5], price_2[:-3]

    return int(uah.replace(" ", "")), int(usd.replace(" ", ""))


def clean(values: list):
    """
    Clear the list from spaces
    """
    return [" ".join(i.split()) for i in values]


def create_dict(keys: list, values: list, mapper: dict):
    return {mapper[key]: value for key, value in zip(keys, values) if key in mapper}


def find_features(data: dict, key_1: str, key_2: str):
    """
    Find features in json object
    """
    features = {}

    for feature in data:
        for key, value in MAP_ADDITIONAL_INFO.items():
            if feature[key_1] == value:
                features[key] = feature[key_2]
    return features


def filter_json(text):
    """
    Filter the response from 
    """
    return text.replace("window.__INITIAL_STATE__=", "").replace(
        ";(function(){var s;(s=document.currentScript||document.scripts[document.scripts.length-1]).parentNode.removeChild(s);}());",
        "",
    )
