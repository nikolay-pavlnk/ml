from sqlalchemy import (
    create_engine,
    Column,
    Boolean,
    String,
    Text,
    Date,
    ARRAY,
    Integer,
    Float,
)
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from domria_scraper.settings import DATABASE


DeclarativeBase = declarative_base()


def db_connect():
    return create_engine(URL(**DATABASE))


def create_table(engine):
    DeclarativeBase.metadata.create_all(engine)


class ApartmentsModel(DeclarativeBase):
    __tablename__ = "apartments"

    id = Column(Integer, primary_key=True)
    title = Column("title", String)
    publishing_date = Column("creation_date", Date)
    price_USD = Column("price_USD", Integer)
    price_UAH = Column("price_UAH", Integer)
    photos_url = Column("photos_url", ARRAY(String))
    description = Column("description", Text)
    street_name = Column("street_name", String)
    city_name = Column("city_name", String)
    district_name = Column("district_name", String)
    square = Column("square", String)
    rooms_count = Column("rooms_count", Integer)
    floor = Column("floor", Integer)
    wall_type = Column("wall_type", String)
    verified_apartment = Column("verified_apartment", Boolean)
    latitude = Column("latitude", String)
    longitude = Column("longitude", String)
    year = Column("construction_year", String)
    heating = Column("heating", String)
    seller = Column("seller", String)
    water = Column("water", String)
    build_character = Column("build_character", ARRAY(String))
    traffic_rating = Column("traffic_rating", Float)
    infrastructure_rating = Column("infrastructure_rating", Float)
    security_rating = Column("security_rating", Float)
    ecology_rating = Column("ecology_rating", Float)
    recreation_area_rating = Column("recreation_area_rating", Float)
    rooms_character = Column("rooms_character", ARRAY(String))
    communications = Column("communications", ARRAY(String))
    driveway = Column("driveway", ARRAY(String))
    position_environment = Column("position_environment", ARRAY(String))
    building_number_str = Column("building_number_str", String)
    type_of_selling = Column("type_of_selling", ARRAY(String))
    position_environment = Column("position_environment", ARRAY(String))
    floors_count = Column("floors_count", Integer)
