import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

MODEL_PATH = Path(
    "notebooks/random_forest_flight_price_model_compressed.pkl"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_daypart(departure_time):
    """Convert departure time into the daypart used by the model."""

    hour = departure_time.hour

    if hour < 6:
        return "night"
    elif hour < 12:
        return "morning"
    elif hour < 18:
        return "afternoon"
    else:
        return "evening"


def create_flight_dataframe(
    airline,
    source_airport,
    destination_airport,
    stops,
    days_left,
    departure_daypart,
    departure_day_of_week,
    duration_minutes
):
    """Create the dataframe expected by the trained model."""

    return pd.DataFrame({
        "airline": [airline],
        "source_airport": [source_airport],
        "destination_airport": [destination_airport],
        "stops": [stops],
        "days_left": [days_left],
        "departure_daypart": [departure_daypart],
        "departure_day_of_week": [departure_day_of_week],
        "duration_minutes": [duration_minutes]
    })


def analyze_booking_window(
    airline,
    source_airport,
    destination_airport,
    travel_date,
    stops,
    duration_minutes,
    departure_time
):
    """Evaluate predicted prices from 1 to 180 days before departure."""

    today = pd.Timestamp.today().normalize()

    travel_date = pd.Timestamp(travel_date)

    current_days_left = (travel_date - today).days

    if current_days_left <= 0:
        raise ValueError(
            "Travel date must be in the future."
        )

    # We analyze up to 180 days before departure,
    # but never beyond the current number of days remaining.
    max_days = min(current_days_left, 180)

    booking_days = list(range(1, max_days + 1))

    departure_day_of_week = (
        travel_date.day_name().lower()
    )

    departure_daypart = get_daypart(departure_time)

    # Create one row for every booking scenario.
    flights = pd.DataFrame({
        "airline": [airline] * len(booking_days),
        "source_airport": [source_airport] * len(booking_days),
        "destination_airport": [destination_airport] * len(booking_days),
        "stops": [stops] * len(booking_days),
        "days_left": booking_days,
        "departure_daypart": [
            departure_daypart
        ] * len(booking_days),
        "departure_day_of_week": [
            departure_day_of_week
        ] * len(booking_days),
        "duration_minutes": [
            duration_minutes
        ] * len(booking_days)
    })

    # Predict all scenarios at once.
    predictions = model.predict(flights)

    results = pd.DataFrame({
        "days_left": booking_days,
        "predicted_price": predictions
    })

    # Find minimum predicted price.
    best_index = results["predicted_price"].idxmin()

    best_row = results.loc[best_index]

    # Find today's predicted price if today
    # is inside the 180-day analysis window.
    today_prediction = results[
        results["days_left"] == current_days_left
    ]

    if not today_prediction.empty:
        current_price = today_prediction[
            "predicted_price"
        ].iloc[0]
    else:
        current_price = None

    # Difference from the best predicted price.
    results["difference_from_best"] = (
        results["predicted_price"]
        - best_row["predicted_price"]
    )

    return (
        results,
        best_row,
        current_days_left,
        current_price
    )


# ============================================================
# 4. APPLICATION TITLE
# ============================================================

st.title("✈️ Flight Price Predictor")

st.write(
    """
    Estimate a flight price and discover when the model
    predicts the best time to book.
    """
)

st.info(
    """
    This tool provides estimates based on historical flight data.
    Predictions are not guarantees of future prices.
    """
)


# ============================================================
# 5. USER INPUTS
# ============================================================

st.subheader("Flight information")

col1, col2 = st.columns(2)

with col1:

    airline = st.text_input(
        "Airline",
        value="Vueling"
    )

    source_airport = st.text_input(
        "Origin airport",
        value="BCN"
    )

    destination_airport = st.text_input(
        "Destination airport",
        value="FCO"
    )

with col2:

    travel_date = st.date_input(
        "Travel date"
    )

    stops = st.number_input(
        "Number of stops",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )

    duration_minutes = st.number_input(
        "Flight duration (minutes)",
        min_value=30,
        max_value=2000,
        value=120,
        step=10
    )

departure_time = st.time_input(
    "Departure time"
)


# ============================================================
# 6. PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔎 Analyze Flight",
    type="primary",
    use_container_width=True
)


# ============================================================
# 7. RUN ANALYSIS
# ============================================================

if predict_button:

    try:

        (
            results,
            best,
            current_days,
            current_price
        ) = analyze_booking_window(
            airline=airline,
            source_airport=source_airport.upper(),
            destination_airport=destination_airport.upper(),
            travel_date=travel_date,
            stops=stops,
            duration_minutes=duration_minutes,
            departure_time=departure_time
        )

        # ----------------------------------------------------
        # Main results
        # ----------------------------------------------------

        st.divider()

        st.subheader("📊 Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Best estimated price",
                f"€{best['predicted_price']:.2f}"
            )

        with col2:

            st.metric(
                "Recommended booking window",
                f"{int(best['days_left'])} days"
            )

        st.success(
            f"""
            According to the model, the lowest estimated price
            occurs approximately **{int(best['days_left'])} days
            before departure**.
            """
        )

        # ----------------------------------------------------
        # Current price
        # ----------------------------------------------------

        if current_price is not None:

            difference = (
                current_price
                - best["predicted_price"]
            )

            st.subheader("💰 Booking today")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Estimated price today",
                    f"€{current_price:.2f}"
                )

            with col2:

                st.metric(
                    "Difference vs. best estimate",
                    f"€{difference:.2f}"
                )

        else:

            st.info(
                f"""
                The flight is currently **{current_days} days
                away**, which is outside the 180-day analysis
                window. The model will still evaluate the next
                180 booking scenarios.
                """
            )

        # ----------------------------------------------------
        # Booking window chart
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📈 Estimated Price by Booking Time"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.plot(
            results["days_left"],
            results["predicted_price"],
            linewidth=2
        )

        ax.scatter(
            best["days_left"],
            best["predicted_price"],
            s=100,
            zorder=5,
            label="Best estimated price"
        )

        ax.annotate(
            (
                f"{int(best['days_left'])} days "
                f"→ €{best['predicted_price']:.2f}"
            ),
            xy=(
                best["days_left"],
                best["predicted_price"]
            ),
            xytext=(10, 15),
            textcoords="offset points"
        )

        ax.set_xlabel(
            "Days Before Departure"
        )

        ax.set_ylabel(
            "Predicted Price (€)"
        )

        ax.set_title(
            "Estimated Flight Price by Booking Time"
        )

        ax.grid(
            alpha=0.3
        )

        ax.legend()

        st.pyplot(
            fig,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Results table
        # ----------------------------------------------------

        with st.expander(
            "View detailed predictions"
        ):

            display_results = results.copy()

            display_results[
                "predicted_price"
            ] = display_results[
                "predicted_price"
            ].round(2)

            display_results[
                "difference_from_best"
            ] = display_results[
                "difference_from_best"
            ].round(2)

            st.dataframe(
                display_results,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Unable to generate prediction: {e}"
        )
