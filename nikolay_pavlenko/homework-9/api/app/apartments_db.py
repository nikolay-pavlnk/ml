from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
import sqlalchemy
from settings import DATABASE, table


class ApartmentsDB:
    def __init__(self):
        engine = create_engine(URL(**DATABASE))
        self.session = sessionmaker(bind=engine)
        meta = sqlalchemy.MetaData()
        meta.reflect(bind=engine)
        self.table = meta.tables.get(table)

    @contextmanager
    def session_scope(self):
        session = self.session()
        try:
            yield session
            session.commit()
        finally:
            session.close()

    def get_statistics(self):
        with self.session_scope() as s:
            num_apart = s.query(func.count(self.table.c.id)).scalar()
            min_price_uah = s.query(func.min(self.table.c.price_UAH)).scalar()
            max_price_uah = s.query(func.max(self.table.c.price_UAH)).scalar()
            min_price_usd = s.query(func.min(self.table.c.price_USD)).scalar()
            max_price_usd = s.query(func.max(self.table.c.price_USD)).scalar()
            average_price_usd = int(s.query(func.avg(self.table.c.price_USD)).scalar())
            average_price_uah = int(s.query(func.avg(self.table.c.price_UAH)).scalar())
            max_floor = s.query(func.max(self.table.c.floor)).scalar()
            min_floor = s.query(func.min(self.table.c.floor)).scalar()
            mean_floor = int(s.query(func.avg(self.table.c.floor)).scalar())
            max_rooms = s.query(func.max(self.table.c.rooms_count)).scalar()
            mean_rooms = int(s.query(func.avg(self.table.c.rooms_count)).scalar())
            mean_ecology_rat = int(
                s.query(func.avg(self.table.c.ecology_rating)).scalar()
            )
            mean_traffic_rat = int(
                s.query(func.avg(self.table.c.traffic_rating)).scalar()
            )
            mean_infrastructure_rat = int(
                s.query(func.avg(self.table.c.infrastructure_rating)).scalar()
            )
            mean_security_rat = int(
                s.query(func.avg(self.table.c.security_rating)).scalar()
            )

        return {
            "number_of_apartments": num_apart,
            "number_of_features": len(self.table.columns) - 1,
            "min_price_uah": min_price_uah,
            "max_price_uah": max_price_uah,
            "min_price_usd": min_price_usd,
            "max_price_usd": max_price_usd,
            "average_price_usd": average_price_usd,
            "average_price_uah": average_price_uah,
            "max_floor": max_floor,
            "min_floor": min_floor,
            "mean_floor": mean_floor,
            "max_rooms": max_rooms,
            "mean_rooms": mean_rooms,
            "mean_ecology_rating": mean_ecology_rat,
            "mean_traffic_rating": mean_traffic_rat,
            "mean_infrastructure_rating": mean_infrastructure_rat,
            "mean_security_rating": mean_security_rat,
        }

    def get_records(self, limit, offset):
        with self.session_scope() as s:
            aparts = (
                s.query(self.table)
                .order_by(self.table.c.creation_date)
                .offset(offset)
                .limit(limit)
                .all()
            )

        return [dict(apart) for apart in aparts]
