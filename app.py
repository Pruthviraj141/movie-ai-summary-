import os
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# === 🔐 API KEYS ===
OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# === 1️⃣ Fetch from OMDb ===
def fetch_omdb(title):
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": title, "plot": "full"}
    response = requests.get(url, params=params)
    return response.json()

# === 2️⃣ Fetch similar movies from IMDb ===
def fetch_imdb_comparisons(genre="Drama"):
    url = "https://imdb236.p.rapidapi.com/api/imdb/search"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "imdb236.p.rapidapi.com"
    }
    params = {
        "type": "movie",
        "genre": genre,
        "rows": 5,
        "sortOrder": "ASC",
        "sortField": "id"
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# === 3️⃣ Streaming availability ===
def fetch_streaming_platforms(title):
    url = "https://streaming-availability.p.rapidapi.com/search/title"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "streaming-availability.p.rapidapi.com"
    }
    params = {"title": title, "country": "us"}
    response = requests.get(url, headers=headers, params=params)
    platforms = []
    try:
        results = response.json().get("result", [])
        if results:
            services = results[0].get("streamingInfo", {}).get("us", {})
            for provider, data in services.items():
                link = data.get("link", "")
                platforms.append(f"[{provider.title()}]({link})")
    except Exception as e:
        st.error(f"Streaming API error: {e}")
    return platforms or ["❌ Not available on major platforms."]

# === 4️⃣ LLM Summary ===
def summarize_with_groq(movie, comparisons, streaming_info):
    chat = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5)

    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a movie assistant who gives detailed and fun movie summaries. Include a unique fact, a USP, and streaming info."
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        "🎬 Title: {title}\n📅 Year: {year}\n⭐ Rating: {rating}\n🧾 Plot: {plot}\n"
        "📺 Streaming on:\n{streaming}\n\n"
        "🎥 Similar movies: {comparisons}\n\n"
        "Give me a fun summary, a unique thing about this movie, and a fun fact. Include platform names and links."
    )
    full_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    comparison_titles = ", ".join([
        item['title']['title']
        for item in comparisons.get("results", [])
        if item.get("title") and item["title"].get("title")
    ])

    prompt_input = full_prompt.format_prompt(
        title=movie.get("Title", "N/A"),
        year=movie.get("Year", "N/A"),
        rating=movie.get("imdbRating", "N/A"),
        plot=movie.get("Plot", "N/A"),
        comparisons=comparison_titles or "None found",
        streaming="\n".join(streaming_info)
    ).to_messages()

    return chat.invoke(prompt_input).content

# === 🖥️ Streamlit UI ===
st.set_page_config(page_title="🎬 Movie Summarizer", layout="centered")
st.title("🎬 Movie AI Summarizer")

movie_title = st.text_input("Enter a movie title")

if st.button("Generate Summary") and movie_title:
    with st.spinner("Fetching data..."):
        omdb_data = fetch_omdb(movie_title)
        if omdb_data.get("Response") != "True":
            st.error("❌ Movie not found. Try another title.")
        else:
            genre = omdb_data.get("Genre", "Drama").split(",")[0].strip()
            imdb_data = fetch_imdb_comparisons(genre)
            streaming_data = fetch_streaming_platforms(movie_title)
            result = summarize_with_groq(omdb_data, imdb_data, streaming_data)

            st.markdown("## 📃 Summary")
            st.markdown(result)
