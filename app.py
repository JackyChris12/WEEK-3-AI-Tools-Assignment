import bz2
import streamlit as st

st.set_page_config(page_title="Amazon Reviews NLP", layout="wide")

st.title("Amazon Reviews — NER & Simple Sentiment")

st.markdown(
    "This app wraps the existing `nlp_analysis.py` functions to provide a small web UI for: \n"
    "- rule-based sentiment classification (Positive/Negative/Neutral) \n"
    "- named-entity recognition (PRODUCT, ORG, PERSON, GPE)"
)


@st.cache_resource
def load_modules():
    # Import lazily so Streamlit starts even if heavy modules take time
    # Use cache_resource because we return a module (not pickle-serializable)
    import nlp_analysis as nlp_mod

    return nlp_mod


nlp_mod = load_modules()


def analyze_text(text: str):
    if not text or text.strip() == "":
        return None

    # Sentiment using rule-based function from the repo
    sentiment = nlp_mod.rule_based_sentiment(text)

    # Run spaCy NER (limit length for performance)
    doc = nlp_mod.nlp(text[:2000])
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG", "PERSON", "GPE"]]

    return {"sentiment": sentiment, "entities": entities}


with st.form("analyze_form"):
    user_text = st.text_area("Enter review text to analyze", height=200)
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not user_text or user_text.strip() == "":
        st.warning("Please enter some review text to analyze.")
    else:
        result = analyze_text(user_text)
        if result is None:
            st.error("Analysis failed or returned no result.")
        else:
            st.subheader("Sentiment")
            st.info(result["sentiment"]) 

            st.subheader("Named Entities")
            if result["entities"]:
                for ent_text, ent_label in result["entities"]:
                    st.write(f"- {ent_text} ({ent_label})")
            else:
                st.write("No PRODUCT/ORG/PERSON/GPE entities found.")


st.markdown("---")
st.header("Analyze a compressed dataset (optional)")
uploaded = st.file_uploader("Upload a .bz2 file (FastText `.ft.txt.bz2` expected)", type=["bz2"])
num_reviews = st.number_input("Number of reviews to show", min_value=1, max_value=100, value=5)

if uploaded is not None:
    try:
        raw = uploaded.read()
        decompressed = bz2.decompress(raw).decode("utf-8", errors="ignore")
        lines = decompressed.splitlines()

        shown = 0
        for line in lines:
            if shown >= num_reviews:
                break
            # Try to extract review text using helper if available
            try:
                review = nlp_mod.extract_review_text(line)
            except Exception:
                review = line

            if len(review) < 20:
                continue

            res = analyze_text(review)
            st.subheader(f"Review #{shown + 1}")
            st.write(review)
            st.write(f"Sentiment: {res['sentiment']}")
            if res["entities"]:
                st.write("Entities:")
                for et, el in res["entities"]:
                    st.write(f"- {et} ({el})")
            else:
                st.write("Entities: none")

            shown += 1

    except Exception as e:
        st.error(f"Failed to read/decompress uploaded file: {e}")


st.markdown("---")
st.caption("Built on top of the repository's `nlp_analysis.py` — uses spaCy NER and a simple rule-based sentiment function.")
