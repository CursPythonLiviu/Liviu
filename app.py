from flask import Flask, render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

app = Flask(__name__)

# Funcția pentru generarea datelor și analiza statistică
def generate_data():
    np.random.seed(42)

    pacienti = pd.DataFrame({
        'Varsta': np.random.randint(18, 90, size=100),
        'Sistola': np.random.randint(90, 140, size=100),
        'Diastola': np.random.randint(60, 90, size=100),
        'Oxigen': np.random.randint(90, 100, size=100),
        'Temperatura': np.round(np.random.uniform(36.0, 37.5, size=100), 1),
    })

    pacienti['TensiuneArteriala'] = pacienti['Sistola'] / pacienti['Diastola']

    def grad_risc(row):
        risc = 0

        if row['Oxigen'] < 95:
            risc += 1

        if row['TensiuneArteriala'] > 1.0 or row['TensiuneArteriala'] < 0.9:
            risc += 1

        if row['Temperatura'] > 37.0:
            risc += 1

        return risc

    pacienti['GradRisc'] = pacienti.apply(grad_risc, axis=1)

    def interpretare_medicala(grad_risc):
        if grad_risc == 0:
            return 'Risc scăzut'
        elif grad_risc == 1:
            return 'Risc moderat'
        else:
            return 'Risc ridicat'

    pacienti['InterpretareMedicala'] = pacienti['GradRisc'].apply(interpretare_medicala)

    media_varsta = pacienti['Varsta'].mean()
    media_oxigen = pacienti['Oxigen'].mean()
    mediana_temperatura = pacienti['Temperatura'].median()

    return pacienti, media_varsta, media_oxigen, mediana_temperatura

# Ruta pentru pagina principala
@app.route('/')
def index():
    pacienti, media_varsta, media_oxigen, mediana_temperatura = generate_data()

    # Salvare analiza statistică într-un fișier imagine
    sns.pairplot(pacienti, hue='GradRisc', diag_kind='kde')
    plt.savefig('static/analiza.png')
    plt.close()

    return render_template('index.html', pacienti=pacienti,
                           media_varsta=media_varsta, media_oxigen=media_oxigen, mediana_temperatura=mediana_temperatura)

# Funcția pentru antrenarea și evaluarea modelului
def train_and_evaluate_model(pacienti):
    features = pacienti[['Varsta', 'TensiuneArteriala', 'Oxigen', 'Temperatura']]
    target = pacienti['GradRisc']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = GradientBoostingClassifier(random_state=42)
    classifier.fit(X_train_scaled, y_train)

    y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_matrix_result = confusion_matrix(y_test, y_pred)
    classification_report_result = classification_report(y_test, y_pred)

    return accuracy, confusion_matrix_result, classification_report_result

# Ruta pentru analiza modelului
@app.route('/model_analysis')
def model_analysis():
    pacienti, _, _, _ = generate_data()
    accuracy, confusion_matrix_result, classification_report_result = train_and_evaluate_model(pacienti)

    return render_template('model_analysis.html', accuracy=accuracy,
                           confusion_matrix=confusion_matrix_result, classification_report=classification_report_result)

if __name__ == '__main__':
    app.run(debug=True)
