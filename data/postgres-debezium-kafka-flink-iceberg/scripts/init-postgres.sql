-- Habilitar replicação lógica
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_wal_senders = 4;
ALTER SYSTEM SET max_replication_slots = 4;

-- Criar schema de teste
CREATE SCHEMA IF NOT EXISTS cdc_source;

-- Criar tabela de exemplo para CDC
CREATE TABLE IF NOT EXISTS cdc_source.customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    country VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Criar tabela de pedidos
CREATE TABLE IF NOT EXISTS cdc_source.orders (
    id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES cdc_source.customers(id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    status VARCHAR(50) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Habilitar replica identity (necessário para Debezium capturar valores antigos)
ALTER TABLE cdc_source.customers REPLICA IDENTITY FULL;
ALTER TABLE cdc_source.orders REPLICA IDENTITY FULL;

-- Inserir dados de teste
INSERT INTO cdc_source.customers (name, email, phone, country) VALUES
('John Doe', 'john@example.com', '+1234567890', 'USA'),
('Jane Smith', 'jane@example.com', '+1234567891', 'Canada'),
('Bob Wilson', 'bob@example.com', '+1234567892', 'UK');

INSERT INTO cdc_source.orders (customer_id, total_amount, status) VALUES
(1, 150.00, 'COMPLETED'),
(2, 200.00, 'PENDING'),
(1, 75.50, 'SHIPPED');
