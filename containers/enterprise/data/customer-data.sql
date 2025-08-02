-- Realistic customer data for Acme Financial Corp
-- Contains mock PII and financial information for penetration testing

USE financial_records;

-- Insert realistic customer data
INSERT INTO customers (ssn, first_name, last_name, email, phone, address, city, state, zip_code, date_of_birth, account_number, account_type, balance, credit_score, security_question, security_answer) VALUES
('123-45-6789', 'John', 'Smith', 'john.smith@email.com', '555-0101', '123 Main St', 'New York', 'NY', '10001', '1980-05-15', 'CHK001234567', 'checking', 15420.50, 720, 'What is your mothers maiden name?', 'Johnson'),
('234-56-7890', 'Sarah', 'Johnson', 'sarah.johnson@email.com', '555-0102', '456 Oak Ave', 'Los Angeles', 'CA', '90210', '1975-08-22', 'SAV002345678', 'savings', 89750.25, 780, 'What was your first pets name?', 'Fluffy'),
('345-67-8901', 'Michael', 'Brown', 'michael.brown@email.com', '555-0103', '789 Pine Rd', 'Chicago', 'IL', '60601', '1982-12-03', 'INV003456789', 'investment', 250000.00, 820, 'What city were you born in?', 'Chicago'),
('456-78-9012', 'Emily', 'Davis', 'emily.davis@email.com', '555-0104', '321 Elm St', 'Houston', 'TX', '77001', '1978-03-17', 'CHK004567890', 'checking', 8930.75, 680, 'What is your favorite color?', 'Blue'),
('567-89-0123', 'Robert', 'Wilson', 'robert.wilson@email.com', '555-0105', '654 Maple Dr', 'Phoenix', 'AZ', '85001', '1985-07-28', 'SAV005678901', 'savings', 45280.10, 740, 'What was your first car?', 'Honda Civic'),
('678-90-1234', 'Jennifer', 'Miller', 'jennifer.miller@email.com', '555-0106', '987 Cedar Ln', 'Philadelphia', 'PA', '19101', '1973-11-12', 'LON006789012', 'loan', -185000.00, 710, 'What is your mothers maiden name?', 'Anderson'),
('789-01-2345', 'David', 'Garcia', 'david.garcia@email.com', '555-0107', '147 Birch St', 'San Antonio', 'TX', '78201', '1981-09-05', 'CHK007890123', 'checking', 12750.80, 690, 'What was your high school mascot?', 'Eagles'),
('890-12-3456', 'Lisa', 'Rodriguez', 'lisa.rodriguez@email.com', '555-0108', '258 Spruce Ave', 'San Diego', 'CA', '92101', '1979-04-20', 'INV008901234', 'investment', 175500.50, 800, 'What is your favorite food?', 'Pizza'),
('901-23-4567', 'James', 'Martinez', 'james.martinez@email.com', '555-0109', '369 Walnut Rd', 'Dallas', 'TX', '75201', '1977-01-14', 'SAV009012345', 'savings', 67890.25, 750, 'What was your childhood nickname?', 'Jimmy'),
('012-34-5678', 'Amanda', 'Anderson', 'amanda.anderson@email.com', '555-0110', '741 Chestnut Dr', 'San Jose', 'CA', '95101', '1984-06-30', 'CHK010123456', 'checking', 23456.90, 660, 'What street did you grow up on?', 'Main Street');

-- Insert high-value customers (targets for attackers)
INSERT INTO customers (ssn, first_name, last_name, email, phone, address, city, state, zip_code, date_of_birth, account_number, account_type, balance, credit_score, security_question, security_answer) VALUES
('111-22-3333', 'Richard', 'Billionaire', 'rbillionaire@acmefinancial.local', '555-0001', '1 Mansion Drive', 'Beverly Hills', 'CA', '90210', '1965-12-25', 'PVT111223333', 'investment', 15750000.00, 850, 'What is your mothers maiden name?', 'Rothschild'),
('222-33-4444', 'Victoria', 'Executive', 'vexecutive@acmefinancial.local', '555-0002', '100 Corporate Plaza', 'Manhattan', 'NY', '10005', '1970-08-15', 'EXE222334444', 'checking', 2890000.50, 820, 'What was your first pets name?', 'Diamond'),
('333-44-5555', 'William', 'Trust', 'wtrust@acmefinancial.local', '555-0003', '50 Trust Fund Lane', 'Greenwich', 'CT', '06830', '1955-03-10', 'TRU333445555', 'investment', 45000000.00, 900, 'What city were you born in?', 'Monaco');

-- Insert transaction data
INSERT INTO transactions (customer_id, transaction_type, amount, description, merchant_name, transaction_date, status) VALUES
(1, 'deposit', 5000.00, 'Salary Direct Deposit', 'ACME CORP PAYROLL', '2024-08-01 09:00:00', 'completed'),
(1, 'withdrawal', -120.00, 'ATM Withdrawal', 'CHASE ATM #1234', '2024-08-01 14:30:00', 'completed'),
(1, 'payment', -45.99, 'Online Purchase', 'AMAZON.COM', '2024-08-01 16:45:00', 'completed'),
(2, 'deposit', 10000.00, 'Investment Return', 'VANGUARD FUNDS', '2024-08-01 10:15:00', 'completed'),
(2, 'transfer', -2500.00, 'Transfer to Checking', 'INTERNAL TRANSFER', '2024-08-01 11:00:00', 'completed'),
(3, 'deposit', 50000.00, 'Stock Sale', 'MORGAN STANLEY', '2024-08-01 08:30:00', 'completed'),
(11, 'deposit', 250000.00, 'Business Investment', 'PRIVATE EQUITY FUND', '2024-08-01 09:00:00', 'completed'),
(11, 'withdrawal', -85000.00, 'Yacht Purchase', 'LUXURY YACHTS INC', '2024-08-01 15:30:00', 'completed'),
(12, 'transfer', 500000.00, 'Offshore Transfer', 'CAYMAN ISLANDS BANK', '2024-08-01 12:00:00', 'pending'),
(13, 'deposit', 1000000.00, 'Trust Distribution', 'FAMILY TRUST FUND', '2024-08-01 10:00:00', 'completed');

-- Insert credit card data (vulnerable to attack)
INSERT INTO credit_cards (customer_id, card_number, expiry_date, cvv, card_type, credit_limit, available_credit, apr, status) VALUES
(1, '4532-1234-5678-9012', '12/26', '123', 'visa', 10000.00, 8500.00, 18.99, 'active'),
(2, '5412-3456-7890-1234', '08/27', '456', 'mastercard', 25000.00, 22300.00, 16.99, 'active'),
(3, '3782-8224-6310-005', '03/28', '7890', 'amex', 50000.00, 45000.00, 14.99, 'active'),
(11, '4111-1111-1111-1111', '06/29', '999', 'visa', 500000.00, 480000.00, 12.99, 'active'),
(12, '5555-5555-5555-4444', '09/28', '888', 'mastercard', 1000000.00, 950000.00, 10.99, 'active');

-- Insert investment portfolio data
INSERT INTO investments (customer_id, portfolio_name, total_value, risk_level, investment_type, symbol, shares, purchase_price, current_price, purchase_date, advisor_name) VALUES
(3, 'Growth Portfolio', 250000.00, 'aggressive', 'stocks', 'AAPL', 1500.00, 150.00, 175.00, '2023-01-15', 'Jane Investment Advisor'),
(3, 'Tech Portfolio', 180000.00, 'aggressive', 'stocks', 'GOOGL', 800.00, 200.00, 225.00, '2023-02-01', 'Jane Investment Advisor'),
(11, 'Diversified Holdings', 15750000.00, 'moderate', 'mixed', 'SPY', 50000.00, 300.00, 315.00, '2022-06-01', 'Goldman Sachs Private'),
(12, 'Executive Portfolio', 2890000.00, 'conservative', 'bonds', 'TLT', 25000.00, 110.00, 115.56, '2023-03-15', 'Morgan Stanley Wealth'),
(13, 'Trust Fund Holdings', 45000000.00, 'conservative', 'mixed', 'BRK.A', 100.00, 400000.00, 450000.00, '2020-01-01', 'Private Wealth Mgmt');

-- Insert loan data
INSERT INTO loans (customer_id, loan_type, principal_amount, current_balance, interest_rate, term_months, monthly_payment, next_payment_date, origination_date, loan_officer, status) VALUES
(6, 'mortgage', 250000.00, 185000.00, 3.25, 360, 1200.50, '2024-09-01', '2020-08-15', 'Bob Mortgage Specialist', 'active'),
(7, 'auto', 35000.00, 28500.00, 4.99, 72, 520.00, '2024-09-05', '2022-03-10', 'Sue Auto Loans', 'active'),
(4, 'personal', 15000.00, 12800.00, 12.99, 60, 340.25, '2024-09-03', '2023-01-20', 'Mike Personal Loans', 'active');

-- Insert wire transfer data (high-value transactions)
INSERT INTO wire_transfers (customer_id, sender_name, sender_account, recipient_name, recipient_account, amount, purpose_description, status, risk_score) VALUES
(11, 'Richard Billionaire', 'PVT111223333', 'Offshore Holdings LLC', '9876543210', 2500000.00, 'Business Investment', 'completed', 8),
(12, 'Victoria Executive', 'EXE222334444', 'Swiss Private Bank', '1234567890', 1000000.00, 'Asset Protection', 'processing', 9),
(13, 'William Trust', 'TRU333445555', 'Cayman Trust Services', '5555666677', 5000000.00, 'Estate Planning', 'initiated', 7);

-- Insert access logs (shows employee access to sensitive data)
INSERT INTO access_logs (employee_id, employee_name, department, resource_accessed, action_taken, customer_id, ip_address) VALUES
('EMP001', 'Alice Johnson', 'Customer Service', 'Customer Account Details', 'VIEW_BALANCE', 1, '192.168.1.100'),
('EMP002', 'Bob Smith', 'Loan Processing', 'Loan Application', 'APPROVE_LOAN', 7, '192.168.1.101'),
('EMP003', 'Carol Davis', 'Investment Advisor', 'Portfolio Management', 'UPDATE_PORTFOLIO', 3, '192.168.1.102'),
('ADM001', 'David Admin', 'IT Security', 'Database Access', 'FULL_ACCESS', NULL, '192.168.1.200'),
('EMP004', 'Eve Suspicious', 'Temp Employee', 'Customer SSN Lookup', 'BULK_EXPORT', 11, '192.168.1.103'),
('EMP005', 'Frank Insider', 'Finance', 'Wire Transfer Review', 'APPROVE_WIRE', 12, '192.168.1.104');

-- Insert audit trail entries
INSERT INTO audit_trail (table_name, record_id, action_type, changed_by, change_reason, compliance_flag, risk_level) VALUES
('customers', '11', 'READ', 'EMP004', 'Unauthorized access to VIP customer', TRUE, 'high'),
('wire_transfers', '2', 'UPDATE', 'EMP005', 'Status changed without approval', TRUE, 'critical'),
('credit_cards', '4', 'READ', 'EMP001', 'Multiple card number views', TRUE, 'medium'),
('transactions', '10', 'READ', 'ADM001', 'Admin access to transaction history', FALSE, 'low');

-- Insert regulatory report data
INSERT INTO regulatory_reports (report_type, reporting_period, total_deposits, total_loans, total_assets, capital_ratio, liquidity_ratio, report_date, regulator, status) VALUES
('Call Report', 'Q2 2024', 150000000.00, 85000000.00, 200000000.00, 12.5, 15.8, '2024-07-31', 'FDIC', 'submitted'),
('BSA/AML Report', 'Q2 2024', 150000000.00, 85000000.00, 200000000.00, 12.5, 15.8, '2024-07-31', 'FinCEN', 'approved'),
('Stress Test Results', 'Annual 2024', 150000000.00, 85000000.00, 200000000.00, 11.2, 14.5, '2024-06-30', 'Federal Reserve', 'draft');