
from cosmos import DbtDag, ProjectConfig, ProfileConfig, ExecutionConfig
from cosmos.profiles import PostgresUserPasswordProfileMapping
from cosmos.constants import ExecutionMode
from datetime import datetime
import os

# Define o caminho para o projeto dbt
dbt_project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dbt", "my_simple_dbt_project")

# Configuração do Profile usando a conexão do Airflow
profile_config = ProfileConfig(
    profile_name="my_simple_dbt_project",
    target_name="dev",
    profile_mapping=PostgresUserPasswordProfileMapping(
        conn_id="postgres_default",  # Usando a conexão padrão do Postgres no Airflow
        profile_args={"schema": "public"},
    ),
)

# Configuração de Execução
# Usamos ExecutionMode.LOCAL porque instalamos o dbt-core e dbt-postgres no ambiente do Airflow via requirements.txt
execution_config = ExecutionConfig(
    execution_mode=ExecutionMode.LOCAL,
)

my_cosmos_dag = DbtDag(
    project_config=ProjectConfig(dbt_project_path),
    profile_config=profile_config,
    execution_config=execution_config,
    # Parâmetros padrão do DAG
    schedule_interval="@daily",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    dag_id="my_simple_dbt_dag",
    tags=["dbt", "cosmos"],
)
