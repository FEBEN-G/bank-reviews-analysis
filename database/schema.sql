
-- Database Schema for Bank Reviews Analysis
-- Generated on: 2025-12-03 01:32:43

-- Banks Table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL,
    app_name VARCHAR(200) NOT NULL
);

-- Reviews Table (as per assignment requirements)
CREATE TABLE reviews (
    review_id VARCHAR(100) PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id),
    review_text TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_date DATE NOT NULL,
    sentiment_label VARCHAR(20),
    sentiment_score DECIMAL(3,2),
    source VARCHAR(50) DEFAULT 'Google Play Store'
);

-- Sample Data
INSERT INTO banks (bank_name, app_name) VALUES 
('CBE', 'Commercial Bank of Ethiopia (CBE)'),
('BOA', 'Bank of Abyssinia (BOA)'),
('Dashen', 'Dashen Bank');
            