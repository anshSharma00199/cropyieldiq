"""CropYieldIQ Streamlit client. Talks ONLY to the API (never to the model/DB directly).
Run: API_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py"""

import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/") + "/api/v1"
st.set_page_config(page_title="CropYieldIQ", page_icon="🌾", layout="wide")


def call(method, path, **kw):
    headers = kw.pop("headers", {})
    if st.session_state.get("token"):
        headers["Authorization"] = f"Bearer {st.session_state['token']}"
    try:
        r = requests.request(method, API + path, headers=headers, timeout=30, **kw)
    except requests.RequestException:
        st.error("Cannot reach the server. Please try again shortly.")
        return None
    if r.status_code == 429:
        st.warning(f"Too many requests. Try again in {r.headers.get('Retry-After', '60')} seconds.")
    elif r.status_code == 401 and st.session_state.get("token"):
        st.session_state.pop("token", None)
        st.warning("Session expired. Please log in again.")
    elif r.status_code >= 400:
        detail = (
            r.json().get("detail", "Request failed")
            if r.headers.get("content-type", "").startswith("application/json")
            else "Request failed"
        )
        st.error(detail if isinstance(detail, str) else "Please check your inputs.")
    return r if r.status_code < 400 else None


@st.cache_data(ttl=3600)
def meta():
    r = requests.get(API + "/meta", timeout=15)
    return r.json() if r.ok else None


def show_result(res):
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.metric("Recommended crop", res["recommendation"].title())
        st.write(
            f"Top predicted probability = **{res.get('top_probability', res['top3'][0]['probability']):.1%}**  |  "
            f"90% conformal set: **{', '.join(res.get('conformal_set') or [res['recommendation']])}**  |  "
            f"model `{res['model_version']}`"
        )
        st.dataframe(pd.DataFrame(res["top3"]), hide_index=True, width="stretch")
        for w in res["warnings"]:
            st.warning(w)
        for a in res.get("assumptions", []):
            st.caption(a)
    with c2:
        st.subheader("Typical-range check")
        for a in res["explanation"]["advice"]:
            st.markdown(f"- {a}")
        st.dataframe(
            pd.DataFrame(res["explanation"]["feature_status"])[["label", "value", "status", "typical_low", "typical_high"]],
            hide_index=True,
            width="stretch",
        )
    if res.get("weather"):
        w = res["weather"]
        st.info(
            f"Live weather for {w['location']['name']}: {w['temperature']}°C, {w['humidity']}% humidity, "
            f"{w['rainfall']} mm over {w['window_days']} days (source {w['source']}, fetched {w['fetched_at']})."
        )
    f1, f2, _ = st.columns([1, 1, 6])
    if f1.button("👍 Helpful", key=f"y{res['id']}"):
        call("POST", "/feedback", json={"prediction_id": res["id"], "helpful": True}) and st.toast("Thanks!")
    if f2.button("👎 Not helpful", key=f"n{res['id']}"):
        call("POST", "/feedback", json={"prediction_id": res["id"], "helpful": False}) and st.toast("Thanks!")
    st.caption("Decision support only. Not a certified agronomic prescription.")


# ---------- auth ----------
if not st.session_state.get("token"):
    st.title("🌾 CropYieldIQ")
    t1, t2 = st.tabs(["Log in", "Register"])
    with t1:
        e, p = st.text_input("Email", key="le"), st.text_input("Password", type="password", key="lp")
        if st.button("Log in") and (r := call("POST", "/auth/login", json={"email": e, "password": p})):
            st.session_state["token"] = r.json()["access_token"]
            st.rerun()
    with t2:
        e, p = st.text_input("Email", key="re"), st.text_input("Password (10+ chars, letters and digits)", type="password", key="rp")
        if st.button("Create account") and call("POST", "/auth/register", json={"email": e, "password": p}):
            st.success("Account created. Please log in.")
    st.stop()

m = meta()
if not m:
    st.error("Server is starting or unavailable.")
    st.stop()
with st.sidebar:
    me = call("GET", "/auth/me")
    st.write(f"Signed in as **{me.json()['email']}** ({me.json()['role']})" if me else "")
    if st.button("Log out"):
        call("POST", "/auth/logout")
        st.session_state.clear()
        st.rerun()
    st.caption(f"Model {m['model_version']} · test accuracy {m['metrics']['test_accuracy']:.1%} (clean dataset, optimistic)")

st.title("🌾 CropYieldIQ: Smart Crop Advisory")
fr = m["feature_range"]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Manual input", "Live weather", "What-if", "My history", "Yield Forecast", "Crop Planner"])
with tab1:
    c = st.columns(4)
    vals = {
        "N": c[0].number_input("Nitrogen (N)", 0.0, 300.0, 90.0),
        "P": c[1].number_input("Phosphorus (P)", 0.0, 300.0, 42.0),
        "K": c[2].number_input("Potassium (K)", 0.0, 400.0, 43.0),
        "ph": c[3].number_input("Soil pH", 0.0, 14.0, 6.5),
    }
    c = st.columns(3)
    vals |= {
        "temperature": c[0].number_input("Temperature (°C)", -20.0, 60.0, 24.0),
        "humidity": c[1].number_input("Humidity (%)", 0.0, 100.0, 82.0),
        "rainfall": c[2].number_input("Rainfall (mm)", 0.0, 2000.0, 200.0),
    }
    if st.button("Recommend crop", type="primary"):
        r = call("POST", "/predict", json=vals)
        if r:
            st.session_state["last"] = r.json()
    if "last" in st.session_state and st.session_state["last"]["kind"] == "manual":
        show_result(st.session_state["last"])

with tab2:
    c = st.columns(2)
    district, state = c[0].text_input("District", "Bhopal"), c[1].text_input("State", "Madhya Pradesh")
    c = st.columns(4)
    soil = {
        "N": c[0].number_input("N", 0.0, 300.0, 80.0, key="ln"),
        "P": c[1].number_input("P", 0.0, 300.0, 40.0, key="lpp"),
        "K": c[2].number_input("K", 0.0, 400.0, 40.0, key="lk"),
        "ph": c[3].number_input("pH", 0.0, 14.0, 6.5, key="lph"),
    }
    if st.button("Use live weather"):
        r = call("POST", "/predict/live", json={"district": district, "state": state, **soil})
        if r:
            st.session_state["last"] = r.json()
    if "last" in st.session_state and st.session_state["last"]["kind"] == "live":
        show_result(st.session_state["last"])

with tab3:
    feat = st.selectbox("Vary this input", m["features"], index=m["features"].index("rainfall"))
    st.caption("Uses the values from the Manual input tab as the baseline.")
    if st.button("Run what-if"):
        r = call("POST", "/predict/sensitivity", json={"inputs": vals, "feature": feat, "points": 20})
        if r:
            s = r.json()
            st.line_chart(pd.DataFrame(s["curve"]).set_index("value"))
            st.caption(s["note"])

with tab4:
    r = call("GET", "/history?limit=50")
    if r:
        h = r.json()
        st.dataframe(pd.DataFrame(h)[["created_at", "kind", "crop", "probability"]] if h else pd.DataFrame(), use_container_width=True)

with tab5:
    st.subheader("AI-Powered Yield Forecast")
    st.caption("Estimate crop yield using the trained yield model. This is separate from crop recommendation.")

    c1, c2, c3 = st.columns(3)

    with c1:
        yield_state = st.text_input("State", value="Andhra Pradesh", key="yield_state")
        yield_crop = st.selectbox("Crop", ["rice", "maize", "chickpea"], key="yield_crop")
        yield_year = st.number_input("Year", min_value=1900, max_value=2100, value=2017, step=1, key="yield_year")

    with c2:
        yield_area = st.number_input("Area (ha)", min_value=0.01, max_value=10_000_000.0, value=1000.0, step=100.0, key="yield_area")
        yield_lag = st.number_input("Previous yield (kg/ha)", min_value=0.0, value=1372.0, step=10.0, key="yield_lag")

    with c3:
        yield_temp = st.number_input("Mean temperature (°C)", min_value=-20.0, max_value=60.0, value=27.8, step=0.1, key="yield_temp")
        yield_rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=10000.0, value=926.1, step=10.0, key="yield_rain")

        if st.button("Predict yield", type="primary", key="predict_yield"):
            payload = {
                "state": yield_state.strip(),
                "crop": yield_crop,
                "area_ha": yield_area,
                "year": int(yield_year),
                "lag_yield": yield_lag,
                "temp_mean": yield_temp,
                "rainfall": yield_rain,
            }

            if not payload["state"]:
                st.error("Please enter a state.")
            else:
                try:
                    response = call("POST", "/yield/predict", json=payload)

                    if response is not None:
                        if response.status_code == 200:
                            st.session_state["yield_result"] = response.json()
                        else:
                            st.error(f"Yield prediction failed ({response.status_code}): {response.text}")
                except Exception as exc:
                    st.error(f"Unable to get yield prediction: {exc}")

    result = st.session_state.get("yield_result")

    if result:
        st.divider()
        st.subheader("Forecast result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted yield", f"{result['yield_kg_per_ha']:,.2f} kg/ha")
        c2.metric("Lower interval", f"{result['interval_lo']:,.2f} kg/ha")
        c3.metric("Upper interval", f"{result['interval_hi']:,.2f} kg/ha")

        st.markdown("### Prediction range")

        lower = result["interval_lo"]
        predicted = result["yield_kg_per_ha"]
        upper = result["interval_hi"]

        if upper > lower:
            position = (predicted - lower) / (upper - lower)
            position = max(0.0, min(1.0, position))

            st.progress(position)

        st.caption(f"Lower: {lower:,.2f} kg/ha  |  Predicted: {predicted:,.2f} kg/ha  |  Upper: {upper:,.2f} kg/ha")

        st.caption(f"{(1 - result['alpha']) * 100:.0f}% nominal prediction interval · Model: `{result['model_version']}`")

        st.subheader("Model explanation")
        for sentence in result["shap_sentences"]:
            st.markdown(f"- {sentence}")

        for warning in result["warnings"]:
            st.warning(warning)

        st.info(result["disclaimer"])

        if result.get("prediction_id"):
            st.info(f"Prediction ID: **{result['prediction_id']}** (Saved to your account)")

            st.divider()
            st.subheader("Record Actual Harvest Outcome")
            st.warning(
                "Please submit your actual measured harvest yield **only after harvest is complete**. "
                "This data is used retrospectively to evaluate and calibrate the model."
            )

            with st.form(key=f"yield_outcome_form_{result['prediction_id']}"):
                o_actual = st.number_input(
                    "Actual harvest yield (kg/ha)",
                    min_value=0.1,
                    max_value=100000.0,
                    value=float(round(result["yield_kg_per_ha"], 1)),
                    step=10.0,
                    help="Enter your measured harvest yield in kilograms per hectare (strictly positive).",
                )
                o_date = st.date_input(
                    "Harvest date (optional)",
                    value=None,
                    help="The date on which the crop was harvested.",
                )
                o_notes = st.text_area(
                    "Notes (optional)",
                    max_chars=500,
                    placeholder="e.g., Rainfall variation, pest conditions, or management notes...",
                )
                submitted_outcome = st.form_submit_button("Submit Actual Harvest Yield", type="primary")

                if submitted_outcome:
                    payload = {
                        "yield_prediction_id": result["prediction_id"],
                        "actual_yield_kg_per_ha": float(o_actual),
                        "notes": o_notes.strip(),
                    }
                    if o_date is not None:
                        payload["harvest_date"] = o_date.isoformat()

                    resp = call("POST", "/yield/outcome", json=payload)
                    if resp is not None:
                        st.success(
                            f"Harvest outcome successfully recorded (Outcome ID: {resp.json().get('id')})! Thank you for your feedback."
                        )

with tab6:
    st.subheader("Crop-specific soil and weather guidance")
    st.caption(
        "Compares your inputs with dataset-derived typical ranges. This is not a fertilizer-dose or irrigation-schedule prescription."
    )

    planner_crop = st.selectbox(
        "Crop",
        m["crops"],
        key="planner_crop",
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        pn = st.number_input("Nitrogen (N)", 0.0, 300.0, 90.0, key="planner_n")
        pp = st.number_input("Phosphorus (P)", 0.0, 300.0, 42.0, key="planner_p")
        pk = st.number_input("Potassium (K)", 0.0, 400.0, 43.0, key="planner_k")

    with c2:
        ptemp = st.number_input("Temperature (°C)", -20.0, 60.0, 24.0, key="planner_temp")
        phum = st.number_input("Humidity (%)", 0.0, 100.0, 82.0, key="planner_humidity")

    with c3:
        pph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="planner_ph")
        prain = st.number_input("Rainfall (mm)", 0.0, 2000.0, 200.0, key="planner_rainfall")

    if st.button("Get crop guidance", type="primary", key="planner_submit"):
        payload = {
            "crop": planner_crop,
            "inputs": {
                "N": pn,
                "P": pp,
                "K": pk,
                "temperature": ptemp,
                "humidity": phum,
                "ph": pph,
                "rainfall": prain,
            },
        }

        r = call("POST", "/planner", json=payload)
        if r:
            st.session_state["planner_result"] = r.json()

    result = st.session_state.get("planner_result")

    if result:
        st.subheader(f"Training-data comparison for {result['crop'].title()}")

        display_df = pd.DataFrame(result["feature_status"])[["label", "value", "status", "typical_low", "typical_high"]].rename(
            columns={
                "typical_low": "Dataset P10",
                "typical_high": "Dataset P90",
            }
        )

        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
        )

        for advice in result["advice"]:
            st.markdown(f"- {advice}")

        st.caption(result["disclaimer"])
        st.caption(f"Model: `{result['model_version']}`")
