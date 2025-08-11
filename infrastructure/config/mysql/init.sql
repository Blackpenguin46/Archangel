-- Initialize vulnerable databases and users
CREATE DATABASE IF NOT EXISTS wordpress;
CREATE DATABASE IF NOT EXISTS opencart;
CREATE DATABASE IF NOT EXISTS corporate;

-- Create vulnerable users with weak passwords
CREATE USER 'wordpress'@'%' IDENTIFIED BY 'vulnerable123';
CREATE USER 'opencart'@'%' IDENTIFIED BY 'opencart123';
CREATE USER 'admin'@'%' IDENTIFIED BY 'admin';
CREATE USER 'guest'@'%' IDENTIFIED BY '';

-- Grant excessive privileges (vulnerability)
GRANT ALL PRIVILEGES ON wordpress.* TO 'wordpress'@'%';
GRANT ALL PRIVILEGES ON opencart.* TO 'opencart'@'%';
GRANT ALL PRIVILEGES ON corporate.* TO 'admin'@'%';
GRANT SELECT ON *.* TO 'guest'@'%';

-- Create sample vulnerable data
USE corporate;
CREATE TABLE employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50),
    password VARCHAR(50),
    email VARCHAR(100),
    ssn VARCHAR(11),
    salary DECIMAL(10,2)
);

INSERT INTO employees VALUES 
(1, 'admin', 'password123', 'admin@company.com', '123-45-6789', 75000.00),
(2, 'jdoe', 'qwerty', 'john.doe@company.com', '987-65-4321', 65000.00),
(3, 'msmith', 'password', 'mary.smith@company.com', '555-12-3456', 70000.00);

CREATE TABLE financial_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    account_number VARCHAR(20),
    balance DECIMAL(15,2),
    customer_name VARCHAR(100)
);

INSERT INTO financial_data VALUES
(1, '1234567890', 50000.00, 'John Doe'),
(2, '0987654321', 75000.00, 'Mary Smith'),
(3, '5555555555', 100000.00, 'Admin User');

FLUSH PRIVILEGES;