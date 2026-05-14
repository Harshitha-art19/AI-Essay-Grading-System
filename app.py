import streamlit as st
import joblib
from langdetect import detect
from deep_translator import GoogleTranslator

# =========================
# LOAD MODEL & TOOLS
# =========================

model = joblib.load("model/essay_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Essay Grader",
    page_icon="🧠",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>
    🧠 Smart Automated Essay Grading System
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("---")

st.write(
    """
    This AI system can:
    
    ✔ Grade Essays  
    ✔ Explain Scores  
    ✔ Analyze Vocabulary  
    ✔ Detect Multiple Languages  
    """
)

# =========================
# ESSAY INPUT
# =========================

essay = st.text_area(
    "✍️ Enter Your Essay",
    height=300,
    placeholder="Write your essay here..."
)

# =========================
# ANALYZE BUTTON
# =========================

if st.button("🚀 Analyze Essay"):

    if essay.strip() == "":
        st.error("Please enter an essay.")

    else:

        # =========================
        # LANGUAGE DETECTION
        # =========================

        try:
            detected_language = detect(essay)

        except:
            detected_language = "unknown"

        language_map = {
            "en": "English",
            "hi": "Hindi",
            "te": "Telugu"
        }

        language_name = language_map.get(
            detected_language,
            detected_language
        )

        st.subheader("🌍 Language Detection")

        st.info(f"Detected Language: {language_name}")

        # =========================
        # TRANSLATION TO ENGLISH
        # =========================

        translated_essay = essay

        if detected_language != "en":

            try:

                translated_essay = GoogleTranslator(
                    source='auto',
                    target='en'
                ).translate(essay)

                st.subheader("🌐 Translated Essay")

                st.write(translated_essay)

            except:
                st.warning("Translation failed.")

        # =========================
        # VECTORIZE ESSAY
        # =========================

        essay_vector = vectorizer.transform([translated_essay])

        # =========================
        # PREDICT SCORE
        # =========================

        prediction = model.predict(essay_vector)[0]

        score = round(prediction, 2)

        # =========================
        # BASIC STATISTICS
        # =========================

        words = len(translated_essay.split())

        characters = len(translated_essay)

        # =========================
        # VOCABULARY ANALYSIS
        # =========================

        unique_words = len(
            set(translated_essay.lower().split())
        )

        if words > 0:
            lexical_diversity = unique_words / words
        else:
            lexical_diversity = 0

        # =========================
        # SENTENCE ANALYSIS
        # =========================

        sentences = translated_essay.split('.')

        if len(sentences) > 0:
            avg_sentence_length = words / len(sentences)
        else:
            avg_sentence_length = 0

        # =========================
        # DISPLAY SCORE
        # =========================

        st.success(f"🎯 Predicted Score: {score}/10")

        st.subheader("📈 Score Meter")

        st.progress(min(int(score * 10), 100))

        st.write("---")

        # =========================
        # ESSAY STATISTICS
        # =========================

        st.subheader("📊 Essay Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"Words: {words}")

        with col2:
            st.info(f"Characters: {characters}")

        with col3:
            st.info(f"Unique Words: {unique_words}")

        # =========================
        # VOCABULARY LEVEL ANALYZER
        # =========================

        st.write("---")

        st.subheader("📚 Vocabulary Analysis")

        if lexical_diversity > 0.75:
            vocab_level = "Advanced"

        elif lexical_diversity > 0.45:
            vocab_level = "Intermediate"

        else:
            vocab_level = "Beginner"

        if vocab_level == "Advanced":
            st.success(f"Vocabulary Level: {vocab_level}")

        elif vocab_level == "Intermediate":
            st.info(f"Vocabulary Level: {vocab_level}")

        else:
            st.warning(f"Vocabulary Level: {vocab_level}")

        st.write(f"Vocabulary Richness Score: {round(lexical_diversity, 2)}")

        # =========================
        # AI FEEDBACK
        # =========================

        st.write("---")

        st.subheader("📝 AI Feedback")

        if words < 50:
            st.warning(
                "Essay is too short. Add more detailed explanation."
            )

        elif words > 300:
            st.success(
                "Good essay length with detailed content."
            )

        if score >= 8:
            st.success(
                "Excellent essay with strong clarity and vocabulary."
            )

        elif score >= 6:
            st.info(
                "Good essay. Improve structure and explanation."
            )

        else:
            st.error(
                "Essay needs improvement in quality and organization."
            )

        # =========================
        # AI EXPLANATION SYSTEM
        # =========================

        st.write("---")

        st.subheader("🧠 Why This Score Was Given")

        if unique_words > 40:
            st.success(
                "✔ Strong vocabulary usage detected."
            )

        else:
            st.warning(
                "❌ Vocabulary is limited. Try richer words."
            )

        if words > 100:
            st.success(
                "✔ Essay length is appropriate."
            )

        else:
            st.warning(
                "❌ Essay is too short for deep evaluation."
            )

        if avg_sentence_length > 10:
            st.success(
                "✔ Sentences are reasonably detailed."
            )

        else:
            st.warning(
                "❌ Sentences are too small and simple."
            )

        # =========================
        # OVERALL EXPLANATION
        # =========================

        st.subheader("📌 Overall AI Explanation")

        if score >= 8:
            st.info(
                """
                The essay received a high score because it demonstrates
                strong vocabulary, good structure, and clear explanation.
                """
            )

        elif score >= 6:
            st.info(
                """
                The essay is reasonably good but can improve in vocabulary,
                structure, and detailed explanation.
                """
            )

        else:
            st.info(
                """
                The essay received a lower score due to weak vocabulary,
                limited structure, or weak organization.
                """
            )

        # =========================
        # FINAL GRADE
        # =========================

        st.write("---")

        st.subheader("🏆 Final Grade")

        if score >= 8:
            st.success("Grade: A")

        elif score >= 6:
            st.info("Grade: B")

        elif score >= 4:
            st.warning("Grade: C")

        else:
            st.error("Grade: D")