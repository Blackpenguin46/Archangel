-- Acme Financial Corp Database Schema
-- Contains realistic financial data structures with mock sensitive information

CREATE DATABASE IF NOT EXISTS financial_records;
USE financial_records;

-- Customer accounts with PII
CREATE TABLE customers (
    customer_id INT PRIMARY KEY AUTO_INCREMENT,
    ssn VARCHAR(11) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    phone VARCHAR(15),
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(2),
    zip_code VARCHAR(10),
    date_of_birth DATE,
    account_number VARCHAR(20) UNIQUE,
    account_type ENUM('checking', 'savings', 'investment', 'loan'),
    balance DECIMAL(15,2),
    credit_score INT,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    security_question VARCHAR(200),
    security_answer VARCHAR(100)
);

-- Transaction records with financial data
CREATE TABLE transactions (
    transaction_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    transaction_type ENUM('deposit', 'withdrawal', 'transfer', 'payment', 'fee'),
    amount DECIMAL(15,2) NOT NULL,
    description VARCHAR(200),
    merchant_name VARCHAR(100),
    merchant_category VARCHAR(50),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    routing_number VARCHAR(9),
    account_number VARCHAR(20),
    reference_number VARCHAR(50),
    status ENUM('pending', 'completed', 'failed', 'reversed'),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Credit card information
CREATE TABLE credit_cards (
    card_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    card_number VARCHAR(19) NOT NULL,  -- Encrypted in real system
    expiry_date VARCHAR(5),
    cvv VARCHAR(4),
    card_type ENUM('visa', 'mastercard', 'amex', 'discover'),
    credit_limit DECIMAL(15,2),
    available_credit DECIMAL(15,2),
    apr DECIMAL(5,2),
    issue_date DATE,
    status ENUM('active', 'suspended', 'closed'),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Investment portfolios
CREATE TABLE investments (
    investment_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    portfolio_name VARCHAR(100),
    total_value DECIMAL(15,2),
    risk_level ENUM('conservative', 'moderate', 'aggressive'),
    investment_type VARCHAR(50),
    symbol VARCHAR(10),
    shares DECIMAL(15,4),
    purchase_price DECIMAL(10,2),
    current_price DECIMAL(10,2),
    purchase_date DATE,
    advisor_name VARCHAR(100),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Loan information
CREATE TABLE loans (
    loan_id INT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    loan_type ENUM('mortgage', 'auto', 'personal', 'business', 'student'),
    principal_amount DECIMAL(15,2),
    current_balance DECIMAL(15,2),
    interest_rate DECIMAL(5,2),
    term_months INT,
    monthly_payment DECIMAL(10,2),
    next_payment_date DATE,
    origination_date DATE,
    collateral_description VARCHAR(200),
    loan_officer VARCHAR(100),
    status ENUM('active', 'paid_off', 'default', 'delinquent'),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Internal employee access logs
CREATE TABLE access_logs (
    log_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    employee_id VARCHAR(20),
    employee_name VARCHAR(100),
    department VARCHAR(50),
    access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resource_accessed VARCHAR(200),
    action_taken VARCHAR(100),
    customer_id INT,
    ip_address VARCHAR(45),
    user_agent TEXT,
    session_id VARCHAR(100),
    success BOOLEAN DEFAULT TRUE
);

-- Compliance and audit trail
CREATE TABLE audit_trail (
    audit_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    table_name VARCHAR(50),
    record_id VARCHAR(50),
    action_type ENUM('CREATE', 'READ', 'UPDATE', 'DELETE'),
    old_values JSON,
    new_values JSON,
    changed_by VARCHAR(100),
    change_reason VARCHAR(200),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    compliance_flag BOOLEAN DEFAULT FALSE,
    risk_level ENUM('low', 'medium', 'high', 'critical')
);

-- Regulatory reporting data
CREATE TABLE regulatory_reports (
    report_id INT PRIMARY KEY AUTO_INCREMENT,
    report_type VARCHAR(50),
    reporting_period VARCHAR(20),
    total_deposits DECIMAL(20,2),
    total_loans DECIMAL(20,2),
    total_assets DECIMAL(20,2),
    risk_weighted_assets DECIMAL(20,2),
    capital_ratio DECIMAL(5,2),
    liquidity_ratio DECIMAL(5,2),
    report_date DATE,
    submitted_date TIMESTAMP,
    regulator VARCHAR(50),
    status ENUM('draft', 'submitted', 'approved', 'rejected')
);

-- Wire transfer records (high-value transactions)
CREATE TABLE wire_transfers (
    wire_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    customer_id INT,
    sender_name VARCHAR(100),
    sender_account VARCHAR(20),
    sender_routing VARCHAR(9),
    sender_bank VARCHAR(100),
    recipient_name VARCHAR(100),
    recipient_account VARCHAR(20),
    recipient_routing VARCHAR(9),
    recipient_bank VARCHAR(100),
    amount DECIMAL(15,2),
    currency VARCHAR(3) DEFAULT 'USD',
    exchange_rate DECIMAL(10,6),
    purpose_code VARCHAR(10),
    purpose_description VARCHAR(200),
    origination_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completion_timestamp TIMESTAMP,
    fees DECIMAL(8,2),
    status ENUM('initiated', 'processing', 'completed', 'failed', 'cancelled'),
    risk_score INT,
    aml_flag BOOLEAN DEFAULT FALSE,
    ofac_checked BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes for performance
CREATE INDEX idx_customers_ssn ON customers(ssn);
CREATE INDEX idx_customers_account ON customers(account_number);
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_access_logs_employee ON access_logs(employee_id);
CREATE INDEX idx_access_logs_time ON access_logs(access_time);
CREATE INDEX idx_wire_transfers_amount ON wire_transfers(amount);
CREATE INDEX idx_wire_transfers_date ON wire_transfers(origination_timestamp);

-- Create users with realistic permissions
CREATE USER 'finance_app'@'%' IDENTIFIED BY 'F1n@pp2024!';
GRANT SELECT, INSERT, UPDATE ON financial_records.* TO 'finance_app'@'%';

CREATE USER 'reporting_user'@'%' IDENTIFIED BY 'R3p0rt1ng2024!';
GRANT SELECT ON financial_records.* TO 'reporting_user'@'%';

CREATE USER 'audit_user'@'%' IDENTIFIED BY 'Aud1tU53r2024!';
GRANT SELECT, INSERT ON financial_records.audit_trail TO 'audit_user'@'%';
GRANT SELECT ON financial_records.* TO 'audit_user'@'%';

-- Weak user for penetration testing
CREATE USER 'legacy_app'@'%' IDENTIFIED BY 'password123';
GRANT ALL PRIVILEGES ON financial_records.* TO 'legacy_app'@'%';

FLUSH PRIVILEGES;