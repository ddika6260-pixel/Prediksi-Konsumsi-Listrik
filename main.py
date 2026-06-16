import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, r2_score


# ==========================
# Membaca Dataset
# ==========================

file_path = "dataset/household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values="?",
    low_memory=False
)


print("Dataset berhasil dibaca")


# ==========================
# Data Cleaning
# ==========================

df = df.dropna()


kolom_angka = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]


df[kolom_angka] = df[kolom_angka].astype(float)



# ==========================
# Menentukan Input dan Target
# ==========================

X = df[
    [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
]


# Target yang ingin diprediksi
y = df["Global_active_power"]



# ==========================
# Membagi Data Training dan Testing
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ==========================
# REGRESI LINEAR
# ==========================

regresi = LinearRegression()

regresi.fit(
    X_train,
    y_train
)


prediksi_regresi = regresi.predict(
    X_test
)


print("\n===== HASIL REGRESI LINEAR =====")

print(
    "MAE :",
    mean_absolute_error(
        y_test,
        prediksi_regresi
    )
)


print(
    "R2 :",
    r2_score(
        y_test,
        prediksi_regresi
    )
)



# ==========================
# RANDOM FOREST
# ==========================

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


random_forest.fit(
    X_train,
    y_train
)


prediksi_rf = random_forest.predict(
    X_test
)


print("\n===== HASIL RANDOM FOREST =====")

print(
    "MAE :",
    mean_absolute_error(
        y_test,
        prediksi_rf
    )
)


print(
    "R2 :",
    r2_score(
        y_test,
        prediksi_rf
    )
)



# ==========================
# Prediksi Data Baru
# ==========================

data_baru = [[
    0.5,
    230,
    10,
    0,
    1,
    0
]]


hasil = random_forest.predict(data_baru)


print("\nPrediksi Konsumsi Listrik:")
print(round(hasil[0], 2), "kW")