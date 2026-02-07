import streamlit as st
from rag_engine import ask  

st.set_page_config(page_title="Car Finder RAG", layout="centered")

# Background style
page_bg_css = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://newsroom.mobile.de/wp-content/uploads/2025/04/Gross-mobile_de_pressekit_relaunch_01.jpeg");
    background-size: cover;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Find Your Car</h1>", unsafe_allow_html=True)

query = st.text_input("", placeholder="Tell us what car you want...")

if st.button("Search"):
    if query.strip() == "":
        st.warning("Please enter a search query.")
    else:
        st.markdown("### 🔍 Results")

        result = ask(query)
        st.write(result["reasoning"])

        for car in result["cars"]:
            st.markdown(
                f"""
                <div style='background-color: rgba(0,0,0,0.9);
                            padding: 20px;
                            border-radius: 10px;
                            margin-bottom: 15px;
                            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);'>
                    <h3>{car['model']}</h3>
                    <p><b>Price:</b> {car['price']}</p>
                    <p><b>Mileage:</b> {car['mileage']}</p>
                    <p><b>Fuel:</b> {car['fuel']}</p>
                    <a href='{car['link']}' target='_blank'>View Car →</a>
                </div>
                """,
                unsafe_allow_html=True
            )