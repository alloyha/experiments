#!/usr/bin/env python3
"""
Consume Kafka
Consome e imprime eventos CDC de um tópico Kafka.

Uso:
  TOPIC=customers python3 scripts/consume_kafka.py
  TOPIC=customers MAX_MESSAGES=5 python3 scripts/consume_kafka.py
"""

import json
import os
import sys

from kafka import KafkaConsumer

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.environ.get("TOPIC")
MAX_MESSAGES = os.environ.get("MAX_MESSAGES")
FROM_BEGINNING = os.environ.get("FROM_BEGINNING", "true").lower() != "false"


def main():
    if not TOPIC:
        print("✗ Defina a variável de ambiente TOPIC (ex: TOPIC=customers)")
        sys.exit(1)

    print(f"Conectando ao Kafka ({BOOTSTRAP_SERVERS}), tópico '{TOPIC}'...")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest" if FROM_BEGINNING else "latest",
        enable_auto_commit=False,
        consumer_timeout_ms=10000,
        value_deserializer=lambda v: v.decode("utf-8") if v else None,
    )

    print(f"✓ Aguardando mensagens em '{TOPIC}' (Ctrl+C para sair)...\n")

    count = 0
    limit = int(MAX_MESSAGES) if MAX_MESSAGES else None

    try:
        for message in consumer:
            count += 1
            print(f"--- Mensagem {count} (offset={message.offset}, partition={message.partition}) ---")
            try:
                payload = json.loads(message.value)
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            except (TypeError, ValueError):
                print(message.value)
            print()

            if limit and count >= limit:
                break
    except KeyboardInterrupt:
        print("\n✓ Encerrando consumer...")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
