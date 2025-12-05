import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("columns.json", "r") as f:
    final_columns = json.load(f)



def preprocess_input_df(df):
    df = df.copy()

    # mileage
    if "mileage" in df.columns:
        df["mileage"] = df["mileage"].astype(str).str.extract(r"([\d\.]+)").astype(float)

    # engine
    if "engine" in df.columns:
        df["engine"] = df["engine"].astype(str).str.extract(r"(\d+)").astype(float)

    # max_power
    if "max_power" in df.columns:
        df["max_power"] = df["max_power"].astype(str).str.extract(r"([\d\.]+)").astype(float)

    df = df.fillna(0)
    return df



def prepare_features(df):
    df = preprocess_input_df(df)

    cat_cols = ["fuel", "seller_type", "transmission", "owner", "seats"]

    df_num = df.drop(columns=cat_cols, errors="ignore")
    df_cat = df[cat_cols]

    df_cat_ohe = ohe.transform(df_cat)
    df_cat_ohe = pd.DataFrame(
        df_cat_ohe,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index
    )

    df_final = pd.concat([df_num, df_cat_ohe], axis=1)
    df_final = df_final.reindex(columns=final_columns, fill_value=0)

    return df_final


st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Предсказание стоимости авто (ElasticNet)")


tabs = st.tabs(["📊 EDA", "🔮 Предсказание", "⚖️ Веса модели"])



with tabs[0]:
    st.header("Исследовательский анализ данных")

    uploaded_eda = st.file_uploader("Загрузите CSV для EDA:", type="csv")

    if uploaded_eda is not None:
        df = pd.read_csv(uploaded_eda)

        st.subheader("Первые строки:")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=np.number).columns

        st.subheader("Гистограммы:")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)



with tabs[1]:
    st.header("Сделать предсказание")

    mode = st.radio("Способ:", ["Загрузить CSV", "Ввести вручную"])

    # ----- CSV -----
    if mode == "Загрузить CSV":
        file = st.file_uploader("Загрузите CSV:", type="csv")

        if file is not None:
            df = pd.read_csv(file)

            df_final = prepare_features(df)
            df_scaled = scaler.transform(df_final)
            preds = model.predict(df_scaled)

            df["predicted_price"] = preds

            st.subheader("Предсказания:")
            st.dataframe(df.head())

            st.download_button(
                "Скачать результат",
                df.to_csv(index=False).encode("utf-8"),
                "predictions.csv",
                "text/csv"
            )

  
    else:
        st.subheader("Введите параметры:")

        year = st.number_input("Год выпуска", 1990, 2023, 2015)
        km = st.number_input("Пробег", 0, 300000, 60000)
        mileage = st.number_input("Расход", 5.0, 40.0, 18.0)
        engine = st.number_input("Объём двигателя", 500, 5000, 1200)
        max_power = st.number_input("Мощность", 30.0, 300.0, 80.0)
        name_wc = st.number_input("Количество слов в названии", 1, 10, 3)

        fuel = st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG"])
        seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("КПП", ["Manual", "Automatic"])
        owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        seats = st.selectbox("Сиденья", [4, 5, 6, 7, 8, 9, 10, 14])

        if st.button("Предсказать"):
            df = pd.DataFrame([{
                "year": year,
                "km_driven": km,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "name_word_count": name_wc,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
                "seats": seats
            }])

            df_final = prepare_features(df)
            df_scaled = scaler.transform(df_final)
            pred = model.predict(df_scaled)[0]

            st.success(f"Предсказанная цена: {pred:,.0f} ₹")



with tabs[2]:
    st.header("Веса модели ElasticNet")

    coefs = pd.DataFrame({
        "feature": final_columns,
        "weight": model.coef_
    })

    # ДВА ВАРИАНТА: как есть + абсолютные значения
    coefs["abs_weight"] = coefs["weight"].abs()
    coefs = coefs.sort_values("abs_weight", ascending=False)

    st.subheader("Таблица весов")
    st.dataframe(coefs)

    # Цвета: красные отрицательные, зелёные положительные
    colors = ["green" if w > 0 else "red" for w in coefs["weight"]]

    st.subheader("График весов")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(coefs["feature"], coefs["weight"], color=colors)
    ax.set_title("Веса признаков (зелёные — увеличивают цену, красные — уменьшают)")
    plt.gca().invert_yaxis()
    st.pyplot(fig)
