# 🚗 Find Your Car in Kiel – RAG-Based Recommendation System

## 📌 Overview
This project implements a **Retrieval Augmented Generation (RAG)** based intelligent car recommendation assistant that allows users to search for vehicles using natural language queries.

Inspired by the AI search feature introduced by mobile.de, this project replicates a simplified version of next-generation marketplace search using real scraped vehicle listing data from the Kiel region in Germany.

The system combines semantic retrieval, rule-based filtering, neural reranking, and local LLM reasoning to provide accurate and explainable car recommendations.

---

## 🎯 Motivation
The used car market contains thousands of listings, making it difficult for buyers to find vehicles matching their needs. Traditional marketplace search systems rely heavily on static filters and keyword matching, which fail to capture human intent.

This project aims to bridge this gap by building an AI-powered search assistant that understands natural language queries and recommends relevant vehicles automatically.

---

## 🧠 System Architecture

The RAG pipeline consists of four main stages:

### 1️⃣ Ingestion & Indexing
- Scraped car listing data is processed and cleaned
- Text-based RAG context is generated
- Embeddings are created using SentenceTransformers
- Vector embeddings stored using FAISS

### 2️⃣ Hybrid Retrieval
- Semantic vector search retrieves top matching vehicles
- Regex-based parser extracts strict price constraints
- Metadata filtering ensures numerical accuracy

### 3️⃣ Neural Reranking
- Cross Encoder model refines relevance scoring
- Ensures highly accurate recommendations

### 4️⃣ LLM-Based Generation
- Local LLM (Ollama – Qwen2.5:7B)
- Generates natural language explanation for top 3 recommendations
- Minimizes hallucination using structured prompts

---

## ⚙️ Features
- Natural language vehicle search
- Hybrid retrieval architecture
- Price constraint extraction from user queries
- Neural reranking for precision
- Local LLM integration
- Streamlit-based interactive UI
- Real marketplace dataset

---

## 🛠 Technology Stack

### Programming
- Python

### Machine Learning & NLP
- SentenceTransformers
- CrossEncoder (ms-marco-MiniLM-L-6-v2)

### Vector Search
- FAISS

### LLM Integration
- Ollama
- Qwen2.5:7B Model

### Data Collection
- Selenium
- Undetected Chromedriver

### UI
- Streamlit

---

## 📊 Dataset
Vehicle listing data was scraped from **mobile.de**, Germany’s largest automotive marketplace.

The dataset includes:
- Vehicle model and description
- Price
- Mileage
- Fuel type
- Seller information
- Listing links

Data was cleaned, normalized, and converted into JSONL format for efficient RAG processing.

---

## ▶️ How To Run

### 1. Clone Repository
```bash
git clone https://github.com/Karthik-Paluri/rag-car-recommendation-system.git
cd rag-car-recommendation-system
