-- config/init_db.sql
CREATE DATABASE IF NOT EXISTS elevator_system;
USE elevator_system;

-- 系統狀態表
CREATE TABLE IF NOT EXISTS system_status (
    id INT AUTO_INCREMENT PRIMARY KEY,
    person INT NOT NULL,
    ShoppingTrolley INT NOT NULL,
    ShoppingBasket INT NOT NULL,
    coverage FLOAT NOT NULL,
    elevator_status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 使用者表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 插入預設管理員帳號
INSERT INTO users (username, password, role) 
VALUES ('admin', 'admin', 'admin');  -- 在實際應用中應使用加密密碼
