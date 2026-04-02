import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env 파일에 저장된 환경 변수(DB 접속 정보)를 불러옵니다.
load_dotenv()

def get_db_engine():
    """
    MySQL 데이터베이스 연결을 위한 SQLAlchemy Engine을 생성하여 반환합니다.
    이 엔진은 GE 검증이나 SQL 실행 시 공통으로 사용됩니다.
    """
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    # SQLAlchemy 연결 문자열 (MySQL + pymysql 드라이버 사용)
    # 형식: mysql+pymysql://유저명:비밀번호@호스트:포트/DB이름
    database_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    try:
        # 엔진 생성 (연결 통로 개설)
        engine = create_engine(database_url)
        
        # 실제로 연결이 잘 되는지 테스트 (정상일 때만 메시지 출력)
        with engine.connect() as connection:
            print(f"✅ DB 연결 성공: {db_name}")
        
        return engine
    
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return None

def get_table_samples(engine, table_name, sample_count=10):
    """
    이미지 회의 내용 반영: 테이블에서 랜덤 10개 샘플 추출
    """
    try:
        with engine.connect() as connection:
            # MySQL 기준 랜덤 샘플링
            query = text(f"SELECT * FROM `{table_name}` ORDER BY RAND() LIMIT {sample_count}")
            df = pd.read_sql(query, connection)
            # LLM이 읽기 편하도록 리스트-딕셔너리 형태로 반환
            return df.to_dict(orient='records')
    except Exception as e:
        print(f"⚠️ {table_name} 샘플 추출 실패: {e}")
        return []
    
    
if __name__ == "__main__":
    # 1. DB 연결 테스트
    engine = get_db_engine()
    
    if engine:
        # 2. 샘플 추출 테스트 (실제 테이블 명 하나를 넣어보세요)
        # 회의 내용처럼 10개가 잘 나오는지 확인용입니다.
        test_table = "users"  # 실제 DB에 있는 테이블명으로 변경
        samples = get_table_samples(engine, test_table, 10)
        
        print(f"\n🔍 '{test_table}' 테이블 샘플 데이터 (최대 10개):")
        for i, row in enumerate(samples, 1):
            print(f"{i}: {row}")
    else:
        print("❌ 테스트 실패: 엔진을 생성할 수 없습니다.")