from sqlalchemy import func
from src.src.databases_util.databases_schema import Jobs
from dbconnections.databases_connection import get_db_connection
from src.src.databases_util.configs.database_configs import CUSTOMER_DB_CONFIG
from src.src.utils import db_instances_to_dataframe

CUSTOMER_DB_CONNECTION = get_db_connection(CUSTOMER_DB_CONFIG)

system_identifier_id = 'system_identifier_id'


with CUSTOMER_DB_CONNECTION.session as session:
    my_query = session \
        .query(Jobs.process_code, func.count(Jobs.process_code).label('number_jobs_per_process_code')) \
        .filter(Jobs.system_identifier_id == system_identifier_id) \
        .group_by(Jobs.process_code).all()

my_query_df = db_instances_to_dataframe(my_query)



