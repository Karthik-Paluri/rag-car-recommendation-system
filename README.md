# 🚗 RAG-Based Car Recommendation System

## 📌 Overview
This project implements a **Retrieval Augmented Generation (RAG)** based intelligent car recommendation system. It allows users to query vehicle preferences in natural language and returns personalized car recommendations using semantic retrieval, reranking, filtering, and large language model reasoning.

The system combines vector similarity search, cross-encoder reranking, structured filtering, and LLM-based explanation generation to provide accurate and user-friendly recommendations.

---

## 🎯 Problem Statement
Traditional car recommendation systems rely on static filters and keyword matching, which often fail to understand natural language queries or user intent.

This project addresses this limitation by:

- Understanding user queries using semantic embeddings
- Retrieving relevant car listings using vector search
- Improving ranking accuracy using cross-encoder models
- Applying rule-based filters (e.g., price constraints)
- Generating explainable recommendations using LLM reasoning

---

## 🧠 System Architecture

User Query  
⬇  
Embedding Generation  
⬇  
FAISS Vector Retrieval  
⬇  
Cross Encoder Reranking  
⬇  
Price Constraint Filtering  
⬇  
LLM Recommendation Generation  
⬇  
Streamlit UI Output  

---

## ⚙️ Features

- Natural language vehicle search
- Semantic similarity retrieval using Sentence Transformers
- FAISS vector database for fast retrieval
- Cross-encoder reranking for improved accuracy
- Automatic price constraint extraction from user queries
- LLM-generated reasoning for recommendations
- Interactive Streamlit user interface
- Local LLM inference using Ollama

---

## 🛠 Tech Stack

### Programming
- Python

### Machine Learning / NLP
- SentenceTransformers
- CrossEncoder models

### Retrieval
- FAISS Vector Database

### LLM Integration
- Ollama (Local LLM Inference)
- Qwen 2.5 Model

### UI
- Streamlit

### Data Processing
- Pandas
- NumPy

---

## 📊 Dataset
The dataset was collected by scraping vehicle listing data from online automotive platforms. It includes:

- Car model
- Price
- Mileage
- Fuel type
- Listing links
- Structured metadata

---

## ▶️ How To Run The Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Karthik-Paluri/rag-car-recommendation-system.git
cd rag-car-recommendation-system
