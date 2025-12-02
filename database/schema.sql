-- PostgreSQL Database Schema for Bank Reviews Analysis
-- Task 3 Submission
-- Generated: 2025-12-03 02:03:36

-- 1. Create banks table (as per requirements)
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL,
    app_name VARCHAR(200) NOT NULL
);

-- 2. Create reviews table (as per requirements)
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

-- 3. Sample Data for three Ethiopian banks
INSERT INTO banks (bank_name, app_name) VALUES 
('CBE', 'Commercial Bank of Ethiopia (CBE)'),
('BOA', 'Bank of Abyssinia (BOA)'),
('Dashen', 'Dashen Bank');

-- Verification: This schema supports insertion of 400+ reviews
-- The accompanying Python script demonstrates the insertion
