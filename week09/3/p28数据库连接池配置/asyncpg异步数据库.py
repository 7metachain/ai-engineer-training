import os
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# 基本连接配置，可通过环境变量覆盖
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = int(os.getenv("PG_PORT", "5432"))
DB_NAME = os.getenv("PG_DB", "postgres")
DB_USER = os.getenv("PG_USER", "postgres")
DB_PASSWORD = os.getenv("PG_PASSWORD", "postgres")

DATABASE_URL = (
    f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

async def asyncpg_demo():
    print("== asyncpg 连接演示 ==")
    conn = await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        timeout=60.0,
    )
    try:
        version = await conn.fetchval("SELECT version()")
        print(f"PostgreSQL 版本: {version}")

        # 示例 SELECT 查询（不依赖业务表）：生成 1..5 与平方
        rows = await conn.fetch(
            "SELECT n, n*n AS square FROM generate_series($1::int, $2::int) AS n",
            1,
            5,
        )
        print("asyncpg 查询结果:")
        for r in rows:
            print(f"n={r['n']}, square={r['square']}")

        # 查询系统视图，列出前 5 个数据库名称
        db_rows = await conn.fetch(
            "SELECT datname FROM pg_database ORDER BY datname LIMIT 5"
        )
        print("pg_database 前5个库:", [r[0] for r in db_rows])

        # 查询非系统表（如果有），展示前 5 个
        table_rows = await conn.fetch(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY table_schema, table_name
            LIMIT 5
            """
        )
        if table_rows:
            print("非系统表示例:")
            for r in table_rows:
                print(f"{r['table_schema']}.{r['table_name']}")
        else:
            print("当前数据库没有非系统表或无权限。")
    finally:
        await conn.close()
        print("asyncpg 连接已关闭")


async def sqlalchemy_demo():
    print("== SQLAlchemy 异步连接演示 ==")
    engine = create_async_engine(DATABASE_URL, echo=False)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar_one()
            print(f"PostgreSQL 版本: {version}")

            # 示例 SELECT 查询（不依赖业务表）：生成 1..5 与平方（参数绑定）
            q = text(
                "SELECT n, n*n AS square FROM generate_series(CAST(:start AS int), CAST(:end AS int)) AS n"
            ).bindparams(start=1, end=5)
            result2 = await conn.execute(q)
            rows = result2.mappings().all()
            print("SQLAlchemy 查询结果:")
            for row in rows:
                print(f"n={row['n']}, square={row['square']}")

            # 查询系统视图，列出前 5 个数据库名称
            db_q = text("SELECT datname FROM pg_database ORDER BY datname LIMIT 5")
            db_res = await conn.execute(db_q)
            dbs = db_res.scalars().all()
            print("pg_database 前5个库:", dbs)

            # 查询非系统表（如果有），展示前 5 个
            tables_q = text(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('pg_catalog','information_schema')
                ORDER BY table_schema, table_name
                LIMIT 5
                """
            )
            tables_res = await conn.execute(tables_q)
            rows2 = tables_res.mappings().all()
            if rows2:
                print("非系统表示例:")
                for r in rows2:
                    print(f"{r['table_schema']}.{r['table_name']}")
            else:
                print("当前数据库没有非系统表或无权限。")
    finally:
        await engine.dispose()
        print("SQLAlchemy 引擎已关闭")


async def main():
    await asyncpg_demo()
    await sqlalchemy_demo()


if __name__ == "__main__":
    asyncio.run(main())

