"""Configuration module for Agentic RAG system"""
# ↑ Yeh docstring file ke top par likhi gayi hai
# Iska kaam sirf batana hai ke yeh file ek "Configuration Module" hai
# jo ke Agentic RAG (Retrieval-Augmented Generation) system ke liye use hoti hai.

import os
# ↑ Python ka built-in module 'os' import kiya gaya hai
# Yeh environment variables ko access karne ke liye use hota hai
# (jaise GEMINI_API_KEY ko environment se nikalna)

from dotenv import load_dotenv
# ↑ 'dotenv' ek library hai jo .env file ko read kar ke
# environment variables system mein load kar deta hai
# Matlab agar aapki project folder mein .env file hai
# to uske andar likhi cheezen (jaise GEMINI_API_KEY=xxxx)
# yeh line load kar degi aur aapko os.getenv() se wo value mil jaayegi

from langchain_google_genai import ChatGoogleGenerativeAI
# ↑ LangChain ka ek class import kiya gaya hai jo Google Gemini LLM ko
# LangChain framework ke andar wrap karta hai
# Iske zariye aap directly Gemini LLM ko call kar sakte ho apne project mein


# ---- Environment Variables Load karna ----
load_dotenv()
# ↑ Yeh function call ensure karta hai ke .env file se saari keys
# aur variables load ho jaayein operating system ke environment mein
# Ab os.getenv("GEMINI_API_KEY") actual key return karega
# warna None dega agar .env mein value na ho


# ---- Config Class ----
class Config:
    """
    Yeh ek Config class hai jo poore system ke liye 
    central jagah provide karti hai settings aur constants rakhnay ki.

    Fayda:
    - Aapko har jagah same variables dobara likhne ki zaroorat nahi
    - Agar koi cheez change karni ho (model, chunk size, API key ka naam)
      to aap sirf yahan change karte ho, baqi code automatically update ho jaata hai
    """

    # ---- API Key ----
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # ↑ Yeh line system ki environment variables se "GEMINI_API_KEY" ko read kar rahi hai
    # Agar .env file load ho gayi hai to yahan key store ho jaayegi
    # Agar nahi mili to yeh None return karega (error bhi aa sakta hai baad mein)


    # ---- Model Configuration ----
    LLM_MODEL = "gemini-2.0-flash-lite"
    # ↑ Yeh ek string hai jisme default model ka naam rakha gaya hai
    # Gemini ke multiple models hote hain:
    # - gemini-1.5-flash (fast aur sasta)
    # - gemini-1.5-pro   (slow magar zyada powerful)
    # Agar aapko model change karna ho to sirf yahan update karna hoga


    # ---- Document Processing Parameters ----
    CHUNK_SIZE = 500
    # ↑ Yeh parameter decide karta hai ke ek document ko
    # kitne bade chunks mein todna hai embeddings banane se pehle
    # yahan 500 characters/tokens ka chunk size use ho raha hai

    CHUNK_OVERLAP = 50
    # ↑ Yeh overlap hai jo ensure karta hai ke har chunk mein
    # peechle chunk ka thoda sa hissa include ho
    # Taake context break na ho aur semantic meaning preserve rahe


    # ---- Default URLs ----
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
    ]
    # ↑ Agar user khud koi URL provide na kare documents ingest karne ke liye
    # to system default ke taur pe in dono URLs ko use karega
    # Yeh dono AI aur ML research ke blog posts hain jo test data ke liye useful hain


    # ---- LLM Initialization Method ----
    @classmethod
    def get_llm(cls):
        """
        Yeh method ek helper hai jo Gemini LLM ko initialize karta hai aur return karta hai.

        Step by step explanation:
        1. Sabse pehle environment variable "GOOGLE_API_KEY" ko set kiya jaata hai
           kyunki Gemini ka client isi naam se key expect karta hai.
        2. Phir ChatGoogleGenerativeAI ko call karke ek LangChain model object banaya jaata hai.
        3. Finally woh object return kar diya jaata hai, taake baqi system usko use kar sake.
        """

        # ---- Step 1: Google API key set karna ----
        os.environ["GOOGLE_API_KEY"] = cls.GEMINI_API_KEY
        # ↑ Yahan hum manually environment variable "GOOGLE_API_KEY" set kar rahe hain
        # Google Gemini LLM ko key isi naam se chahiye hoti hai
        # Agar aapne GEMINI_API_KEY ko .env file mein rakha hai
        # to ab woh environment mein "GOOGLE_API_KEY" ke naam se bhi available ho jaayegi

        # ---- Step 2: Gemini LLM initialize karna ----
        return ChatGoogleGenerativeAI(model=cls.LLM_MODEL)
        # ↑ Yeh LangChain ka wrapper call karta hai jo Gemini model ko
        # internally use karega aur ek object return karega
        # Jisko baqi project mein call karke queries run ki jaa sakti hain
