"""
Cria ou recria o banco DuckDB a partir do seed.sql.

Uso:
    pip install duckdb
    python build_db.py
"""

from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "patologias.duckdb"
SQL_PATH = BASE_DIR / "seed.sql"


def build_database() -> None:
    if not SQL_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo SQL não encontrado: {SQL_PATH}"
        )

    sql_script = SQL_PATH.read_text(encoding="utf-8")

    con = duckdb.connect(str(DB_PATH))

    try:
        # Executa seed de forma transacional.
        #
        # Se qualquer CREATE / INSERT / CHECK falhar,
        # evitamos persistir um banco parcialmente construído.
        con.execute("BEGIN TRANSACTION")

        try:
            con.execute(sql_script)
            con.execute("COMMIT")

        except Exception:
            con.execute("ROLLBACK")
            raise

        print(f"\nBanco criado em:")
        print(f"  {DB_PATH}\n")

        print_schema_summary(con)
        print_view_summary(con)
        run_sanity_checks(con)

    finally:
        con.close()


def print_schema_summary(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Descobre as tabelas automaticamente no catálogo DuckDB
    e imprime a quantidade de registros.
    """

    tables = con.execute(
        """
        SELECT table_name
        FROM duckdb_tables()
        WHERE schema_name = 'main'
        ORDER BY table_name
        """
    ).fetchall()

    print("Tabelas")
    print("-" * 52)

    for (table_name,) in tables:
        # table_name vem do catálogo DuckDB, não de input externo.
        count = con.execute(
            f'SELECT COUNT(*) FROM "{table_name}"'
        ).fetchone()[0]

        print(
            f"  {table_name:<36}"
            f"{count:>6} linha(s)"
        )

    print()


def print_view_summary(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Lista views analíticas presentes no banco.
    """

    views = con.execute(
        """
        SELECT view_name
        FROM duckdb_views()
        WHERE schema_name = 'main'
        ORDER BY view_name
        """
    ).fetchall()

    if not views:
        return

    print("Views")
    print("-" * 52)

    for (view_name,) in views:
        count = con.execute(
            f'SELECT COUNT(*) FROM "{view_name}"'
        ).fetchone()[0]

        print(
            f"  {view_name:<36}"
            f"{count:>6} linha(s)"
        )

    print()


def run_sanity_checks(
    con: duckdb.DuckDBPyConnection,
) -> None:
    """
    Executa invariantes estruturais simples.

    Esses testes não verificam validade médica dos dados.
    Eles verificam apenas consistência lógica do modelo.
    """

    checks = {
        "patologias possuem sintomas": """
            SELECT COUNT(*)
            FROM patologia AS p
            WHERE NOT EXISTS (
                SELECT 1
                FROM patologia_sintoma AS ps
                WHERE ps.patologia_id = p.id
            )
        """,

        "apresentações possuem composição": """
            SELECT COUNT(*)
            FROM apresentacao AS a
            WHERE NOT EXISTS (
                SELECT 1
                FROM apresentacao_principio_ativo AS apa
                WHERE apa.apresentacao_id = a.id
            )
        """,

        "eficácias dentro de [0,1]": """
            SELECT COUNT(*)
            FROM apresentacao_alivia
            WHERE eficacia < 0
               OR eficacia > 1
               OR probabilidade_resposta < 0
               OR probabilidade_resposta > 1
        """,

        "riscos dentro de [0,1]": """
            SELECT COUNT(*)
            FROM apresentacao_evento_adverso
            WHERE probabilidade < 0
               OR probabilidade > 1
               OR gravidade_peso < 0
               OR gravidade_peso > 1
               OR peso_clinico < 0
               OR peso_clinico > 1
        """,

        "interações não duplicadas/invertidas": """
            SELECT COUNT(*)
            FROM apresentacao_interacao
            WHERE apresentacao_a_id >= apresentacao_b_id
        """,
    }

    print("Sanity checks")
    print("-" * 52)

    failures = []

    for name, query in checks.items():
        violations = con.execute(query).fetchone()[0]

        status = (
            "OK"
            if violations == 0
            else f"FAIL ({violations})"
        )

        print(f"  {name:<38} {status}")

        if violations:
            failures.append(
                (name, violations)
            )

    print()

    if failures:
        details = ", ".join(
            f"{name}: {count}"
            for name, count in failures
        )

        raise RuntimeError(
            f"Falha nos sanity checks: {details}"
        )


if __name__ == "__main__":
    build_database()
