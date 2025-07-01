import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    a = mo.output.append
    return a, mo, pd


@app.cell
def _():
    import micropip
    return (micropip,)


@app.cell
async def _(micropip):
    await micropip.install("plotly")
    return


@app.cell
def _():
    import plotly.express as px
    import plotly.graph_objects as go
    return (px,)


@app.cell
def _(pd):
    pd.options.plotting.backend = "plotly"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **Base de dados usada**

    https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
    """
    )
    return


@app.cell
def _(a, mo):
    file_dir = mo.notebook_location() / "public" / "data.csv"

    a(mo.md(f"Tentando carregar arquivo {file_dir}"))
    return (file_dir,)


@app.cell
def _(a, file_dir, mo, pd):
    try:
        df = pd.read_csv(file_dir, sep=";")
    except Exception:
        a(mo.md("Erro ao ler arquivo local, tentando ler arquivo remoto"))
        import requests, io

        file = requests.get(file_dir)
        df = pd.read_csv(io.BytesIO(file.content), sep=";")
    return (df,)


@app.cell
def _(df):
    df["Target"] = df["Target"].replace(
        ["Dropout", "Graduate", "Enrolled"], [0, 1, 2]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""#Exploração Inicial da Base""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""##Inspeção básica""")
    return


@app.cell
def _(a, df, mo):
    a(mo.md("###Formato da base"))

    a(f"Linhas: {df.shape[0]}")
    a(f"Colunas: {df.shape[1]}")
    return


@app.cell(hide_code=True)
def _(pd):
    # esta célula é responsável por rotular as variáveis categóricas com base nas informações do dataset.

    YES_NO = {1: "Yes", 0: "No"}

    rotulos = {
        "Marital status": {
            1: "single",
            2: "married",
            3: "widower",
            4: "divorced",
            5: "facto union",
            6: "legally separated",
        },
        "Application mode": {
            1: "1st phase - general contingent",
            2: "Ordinance No. 612/93",
            5: "1st phase - special contingent (Azores Island)",
            7: "Holders of other higher courses",
            10: "Ordinance No. 854-B/99",
            15: "International student (bachelor)",
            16: "1st phase - special contingent (Madeira Island)",
            17: "2nd phase - general contingent",
            18: "3rd phase - general contingent",
            26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
            27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
            39: "Over 23 years old",
            42: "Transfer",
            43: "Change of course",
            44: "Technological specialization diploma holders",
            51: "Change of institution/course",
            53: "Short cycle diploma holders",
            57: "Change of institution/course (International)",
        },
        "Course": {
            33: "Biofuel Production Technologies",
            171: "Animation and Multimedia Design",
            8014: "Social Service (evening attendance)",
            9003: "Agronomy",
            9070: "Communication Design",
            9085: "Veterinary Nursing",
            9119: "Informatics Engineering",
            9130: "Equinculture",
            9147: "Management",
            9238: "Social Service",
            9254: "Tourism",
            9500: "Nursing",
            9556: "Oral Hygiene",
            9670: "Advertising and Marketing Management",
            9773: "Journalism and Communication",
            9853: "Basic Education",
            9991: "Management (evening attendance)",
        },
        "Daytime/evening attendance	": {
            1: "Daytime",
            0: "Evening",
        },
        "Previous qualification": {
            1: "Secondary education",
            2: "Higher education - bachelor's degree",
            3: "Higher education - degree",
            4: "Higher education - master's",
            5: "Higher education - doctorate",
            6: "Frequency of higher education",
            9: "12th year of schooling - not completed",
            10: "11th year of schooling - not completed",
            12: "Other - 11th year of schooling",
            14: "10th year of schooling",
            15: "10th year of schooling - not completed",
            19: "Basic education 3rd cycle (9th/10th/11th year) or equiv.",
            38: "Basic education 2nd cycle (6th/7th/8th year) or equiv.",
            39: "Technological specialization course",
            40: "Higher education - degree (1st cycle)",
            42: "Professional higher technical course",
            43: "Higher education - master (2nd cycle)",
        },
        "Nacionality": {
            1: "Portuguese",
            2: "German",
            6: "Spanish",
            11: "Italian",
            13: "Dutch",
            14: "English",
            17: "Lithuanian",
            21: "Angolan",
            22: "Cape Verdean",
            24: "Guinean",
            25: "Mozambican",
            26: "Santomean",
            32: "Turkish",
            41: "Brazilian",
            62: "Romanian",
            100: "Moldova (Republic of)",
            101: "Mexican",
            103: "Ukrainian",
            105: "Russian",
            108: "Cuban",
            109: "Colombian",
        },
        "Mother's qualification": {
            1: "Secondary Education - 12th Year of Schooling or Eq.",
            2: "Higher Education - Bachelor's Degree",
            3: "Higher Education - Degree",
            4: "Higher Education - Master's",
            5: "Higher Education - Doctorate",
            6: "Frequency of Higher Education",
            9: "12th Year of Schooling - Not Completed",
            10: "11th Year of Schooling - Not Completed",
            11: "7th Year (Old)",
            12: "Other - 11th Year of Schooling",
            14: "10th Year of Schooling",
            18: "General commerce course",
            19: "Basic Education",
            33: "12th Year of Schooling - Not Completed",
            34: "Unknown",
            35: "Can't read or write",
            36: "Can read without having a 4th year of schooling",
            37: "Basic education 1st cycle (4th/5th year) or equiv.",
            38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
            39: "Technological specialization course",
            40: "Higher education - degree (1st cycle)",
            41: "Specialized higher studies course",
            42: "Professional higher technical course",
            43: "Higher Education - Master (2nd cycle)",
            44: "Higher Education - Doctorate (3rd cycle)",
        },
        "Father's qualification": {
            1: "Secondary Education - 12th Year of Schooling or Eq.",
            2: "Higher Education - Bachelor's Degree",
            3: "Higher Education - Degree",
            4: "Higher Education - Master's Degree",
            5: "Higher Education - Doctorate",
            6: "Frequency of Higher Education",
            9: "12th Year of Schooling - Not Completed",
            10: "11th Year of Schooling - Not Completed",
            11: "7th Year (Old)",
            12: "Other - 11th Year of Schooling",
            13: "2nd year complementary high school course",
            14: "10th Year of Schooling",
            18: "General commerce course",
            19: "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",
            20: "Complementary High School Course",
            22: "Technical-professional course",
            25: "Complementary High School Course - not concluded",
            26: "7th year of schooling",
            27: "2nd cycle of the general high school course",
            29: "9th Year of Schooling - Not Completed",
            30: "8th year of schooling",
            31: "General Course of Administration and Commerce",
            33: "Supplementary Accounting and Administration",
            34: "Unknown",
            35: "Can't read or write",
            36: "Can read without having a 4th year of schooling",
            37: "Basic education 1st cycle (4th/5th year) or equiv.",
            38: "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",
            39: "Technological specialization course",
            40: "Higher education - degree (1st cycle)",
            41: "Specialized higher studies course",
            42: "Professional higher technical course",
            43: "Higher Education - Master (2nd cycle)",
            44: "Higher Education - Doctorate (3rd cycle)",
        },
        "Mother's occupation": {
            0: "Student",
            1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
            2: "Specialists in Intellectual and Scientific Activities",
            3: "Intermediate Level Technicians and Professions",
            4: "Administrative staff",
            5: "Personal Services, Security and Safety Workers and Sellers",
            6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
            7: "Skilled Workers in Industry, Construction and Craftsmen",
            8: "Installation and Machine Operators and Assembly Workers",
            9: "Unskilled Workers",
            10: "Armed Forces Professions",
            90: "Other Situation",
            99: "(blank)",
            122: "Health professionals",
            123: "teachers",
            125: "Specialists in information and communication technologies (ICT)",
            131: "Intermediate level science and engineering technicians and professions",
            132: "Technicians and professionals, of intermediate level of health",
            134: "Intermediate level technicians from legal, social, sports, cultural and similar services",
            141: "Office workers, secretaries in general and data processing operators",
            143: "Data, accounting, statistical, financial services and registry-related operators",
            144: "Other administrative support staff",
            151: "personal service workers",
            152: "sellers",
            153: "Personal care workers and the like",
            171: "Skilled construction workers and the like, except electricians",
            173: "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",
            175: "Workers in food processing, woodworking, clothing and other industries and crafts",
            191: "cleaning workers",
            192: "Unskilled workers in agriculture, animal production, fisheries and forestry",
            193: "Unskilled workers in extractive industry, construction, manufacturing and transport",
            194: "Meal preparation assistants",
        },
        "Father's occupation": {
            0: "Student",
            1: "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",
            2: "Specialists in Intellectual and Scientific Activities",
            3: "Intermediate Level Technicians and Professions",
            4: "Administrative staff",
            5: "Personal Services, Security and Safety Workers and Sellers",
            6: "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",
            7: "Skilled Workers in Industry, Construction and Craftsmen",
            8: "Installation and Machine Operators and Assembly Workers",
            9: "Unskilled Workers",
            10: "Armed Forces Professions",
            90: "Other Situation",
            99: "(blank)",
            101: "Armed Forces Officers",
            102: "Armed Forces Sergeants",
            103: "Other Armed Forces personnel",
            112: "Directors of administrative and commercial services",
            114: "Hotel, catering, trade and other services directors",
            121: "Specialists in the physical sciences, mathematics, engineering and related techniques",
            122: "Health professionals",
            123: "teachers",
            124: "Specialists in finance, accounting, administrative organization, public and commercial relations",
            131: "Intermediate level science and engineering technicians and professions",
            132: "Technicians and professionals, of intermediate level of health",
            134: "Intermediate level technicians from legal, social, sports, cultural and similar services",
            135: "Information and communication technology technicians",
            141: "Office workers, secretaries in general and data processing operators",
            143: "Data, accounting, statistical, financial services and registry-related operators",
            144: "Other administrative support staff",
            151: "personal service workers",
            152: "sellers",
            153: "Personal care workers and the like",
            154: "Protection and security services personnel",
            161: "Market-oriented farmers and skilled agricultural and animal production workers",
            163: "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
            171: "Skilled construction workers and the like, except electricians",
            172: "Skilled workers in metallurgy, metalworking and similar",
            174: "Skilled workers in electricity and electronics",
            175: "Workers in food processing, woodworking, clothing and other industries and crafts",
            181: "Fixed plant and machine operators",
            182: "assembly workers",
            183: "Vehicle drivers and mobile equipment operators",
            192: "Unskilled workers in agriculture, animal production, fisheries and forestry",
            193: "Unskilled workers in extractive industry, construction, manufacturing and transport",
            194: "Meal preparation assistants",
            195: "Street vendors (except food) and street service providers",
        },
        "Gender": {
            1: "Male",
            0: "Female",
        },
        "Displaced": YES_NO,
        "Educational special needs": YES_NO,
        "Debtor": YES_NO,
        "Tuition fees up to date": YES_NO,
        "Scholarship holder": YES_NO,
        "International": YES_NO,
        "Target": {
            0: "Dropout",
            1: "Graduate",
            2: "Enrolled",
        },
    }


    def aplicar_rotulos(
        df: pd.DataFrame, rotulos_dict: dict[str, dict[int, str]], inplace=False
    ):
        """
        Aplica rótulos às variáveis categóricas

        Parameters:
        df: DataFrame original
        rotulos_dict: Dicionário com os rótulos
        inplace: Se True, modifica o DataFrame original

        Returns:
        DataFrame com rótulos aplicados
        """

        if inplace:
            df_rotulado = df
        else:
            df_rotulado = df.copy()

        for coluna, mapeamento in rotulos_dict.items():
            if coluna in df_rotulado.columns:
                df_rotulado[coluna] = df_rotulado[coluna].map(mapeamento)

        return df_rotulado
    return aplicar_rotulos, rotulos


@app.cell
def _(a, df, mo, pd, rotulos):
    a(mo.md("###Tipos de dados"))

    numericas = [
        "Previous qualification (grade)",
        "Admission grade",
        "Age at enrollment",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]

    alvos = ["Target"]

    categoricas = [col for col in df.columns if col not in numericas + alvos]


    def _():
        tipos_de_dados = []

        for col, dtype in zip(df.columns, df.dtypes):
            tipo = None
            categorias = None
            if col in alvos:
                tipo = "Alvo"
            elif col in numericas:
                tipo = "Numérica"
            elif col in categoricas:
                tipo = "Categórica"
                categorias = df[col].drop_duplicates().sort_values()

            tipo = {
                "Coluna": col,
                "Tipo": tipo,
                "dtype": dtype,
            }

            if categorias is not None:
                if rotulos.get(col):
                    tipo["Categorias"] = categorias.map(rotulos.get(col))
                else:
                    tipo["Categorias"] = categorias

            tipos_de_dados.append(tipo)

        a(pd.DataFrame(tipos_de_dados))


    _()

    a(
        mo.md(
            "A separação entre dados categóricos e numéricos foi feita manualmente, com base nas informações do dataset fornecido no repositório citado no começo da página"
        )
    )
    return categoricas, numericas


@app.cell
def _(a, df, mo, pd):
    a(mo.md("###Visão geral"))

    a(pd.concat([df.head(5), df.tail(5)]))
    return


@app.cell
def _(a, df, mo):
    a(mo.md("###Valores ausentes"))

    a(df.isna().sum())

    a(
        mo.md(
            "Com base na tabela, podemos concluir que não existem valores nulos ou ausentes na base de dados"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""##Análise Descritiva""")
    return


@app.cell
def _(a, df, mo, numericas):
    a(mo.md("###Estatísticas descritivas para variáveis numéricas"))
    a(df[numericas].describe())
    return


@app.cell
def _(a, aplicar_rotulos, categoricas, df, mo, rotulos):
    a(
        mo.md("""
    ###Distribuição de frequências para variáveis categóricas
    Apenas os 3 maiores valores são exibidos
    """)
    )


    def _():
        rotulado = aplicar_rotulos(df, rotulos)

        for c in categoricas:
            counted_df = (
                rotulado[c].value_counts(normalize=True).head(3) * 100
            ).to_frame()
            a(
                mo.ui.table(
                    counted_df,
                    pagination=False,
                    show_download=False,
                    selection=None,
                )
            )


    _()
    return


@app.cell
def _(a, df, mo):
    a(mo.md("###Verificação de duplicatas"))

    a(df[df.duplicated()])

    a(mo.md("Podemos ver que não existem colunas duplicadas"))
    return


@app.cell
def _(mo):
    mo.md(r"""# Visualizações""")
    return


@app.cell
def _(a, mo):
    a(mo.md("##Gráficos exploratórios"))
    return


@app.cell
def _(a, aplicar_rotulos, categoricas, df, mo, px, rotulos):
    a(mo.md("###Histogramas"))


    def _():
        rotulado = aplicar_rotulos(df, rotulos)

        for c in categoricas:
            fig = px.histogram(rotulado, x=c, title=f"Histograma de {c}")
            a(fig)


    _()
    return


@app.cell
def _(a, aplicar_rotulos, df, px, rotulos):
    def _():
        rotulado = aplicar_rotulos(df, rotulos)

        a(
            px.histogram(
                rotulado["Nacionality"][rotulado["Nacionality"] != "Portuguese"],
                title="Nacionalidade sem contar portugueses",
            )
        )

        a(px.histogram(rotulado["Target"], title="Alvo"))


    _()
    return


@app.cell
def _(a, df, mo, numericas, px):
    a(mo.md("###Boxplots"))


    def _():
        figs = []

        for c in numericas:
            fig = px.box(df, y=c, title=f"{c}")
            # Add all data points to the boxplot for better insight
            # fig.update_traces(boxpoints='all', jitter=0.3)
            figs.append(fig)

        rows = [mo.hstack(figs[i : i + 2]) for i in range(0, len(figs), 2)]

        a(mo.vstack(rows))


    _()
    return


@app.cell
def _(a, categoricas, df, mo, numericas, px):
    a(mo.md("###Heatmaps"))


    def _():
        a(mo.md("Numéricas"))
        numericas_corr = df[numericas + ["Target"]].corr()
        a(px.imshow(numericas_corr))

        a(mo.md("Categóricas"))
        categoricas_corr = df[categoricas + ["Target"]].corr()
        a(px.imshow(categoricas_corr))

        a(mo.md("Ambas"))
        df_corr = df.corr()
        a(px.imshow(df_corr))


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""#Preparação""")
    return


@app.cell
def _(mo):
    mo.md(r"""##Definição do problema""")
    return


@app.cell
def _(mo):
    mo.md(
        rf"""
    ###Variável alvo:
    **Target**

    O problema é formulado como uma tarefa de classificação em três categorias: 

    - **dropout** (desistente)
    - **enrolled** (matriculado)
    - **graduate** (graduado)

    como o resultado alvo é uma de 3 categorias, o problema se trata de uma classificação
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##Pré-processamento
    Com base na página de informações do dataset:
    > **Algum pré processamento dos dados foi realizado?**
    >
    > Nós realizamos um pré processamento de dados rigoroso para filtrar os dados de anomalias, outliers inexplicáveis e valores faltantes.

    Mas como os valores de categorias não foram normalizados, aplicaremos a estratégia de **One Hot Encoding** nas colunas categóricas e **Min Max Scaling** nas colúnas numéricas
    """
    )
    return


@app.cell
def _(df, numericas, pd):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numericas])

    numeric_scaled_df = pd.DataFrame(
        scaled, columns=scaler.get_feature_names_out(numericas)
    )

    mm_df_num = numeric_scaled_df

    mm_df_num
    return (mm_df_num,)


@app.cell
def _(categoricas, df, pd):
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(handle_unknown="ignore")

    encoded = encoder.fit_transform(df[categoricas]).toarray()

    categoricas_ohe = encoder.get_feature_names_out(categoricas)

    categorical_oh_encoded_df = pd.DataFrame(encoded, columns=categoricas_ohe)


    oh_df_cat = categorical_oh_encoded_df

    oh_df_cat
    return (oh_df_cat,)


@app.cell
def _(df, mm_df_num, oh_df_cat, pd):
    # Normalizado

    ndf = pd.concat([mm_df_num, oh_df_cat, df["Target"]], axis=1)
    ndf
    return (ndf,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ###Divisão de treino/teste

    Também seguindo as informações do dataset, podemos ver que o split (divisão) foi definido como: 

    - 80% para treinamento
    - 20% para testes
    """
    )
    return


@app.cell
def _(a, mo, ndf):
    from sklearn.model_selection import train_test_split

    a(mo.md("##Divisão do dataset"))

    X = ndf.drop("Target", axis=1)
    y = ndf["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2
    )

    a(
        mo.md(f"""
    **Base de treinamento:**

    | formato     | features        | alvos           |
    |-------------|-----------------|-----------------|
    | treinamento | {X_train.shape} | {y_train.shape} |
    | teste       | {X_test.shape}  | {y_test.shape}  |

    """)
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
    #Comparação de algoritmos
    ## Algoritmos usados
    """
    )
    return


@app.cell
def _():
    classification_reports = {}
    return (classification_reports,)


@app.cell
def _(px):
    # Confusion matrix

    from sklearn.metrics import confusion_matrix

    values = ["Dropout", "Graduate", "Enrolled"]


    def cm_heatmap(test, pred):
        cm = confusion_matrix(test, pred)

        return px.imshow(
            cm,
            labels={"x": "Previsto", "y": "Verdadeiro"},
            x=values,
            y=values,
            text_auto=True,
        )
    return cm_heatmap, values


@app.cell
def _(pd):
    from sklearn.metrics import accuracy_score, classification_report


    def cr_table(test, pred):
        cr = classification_report(test, pred, output_dict=True)
        crdf = pd.DataFrame.from_dict(cr)

        return crdf
    return (cr_table,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Naive Bayes

    Modelo probabilístico simples e rápido, assume independência entre variáveis.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # def _():
    #     from sklearn.naive_bayes import GaussianNB, CategoricalNB
    #     from sklearn.metrics import log_loss
    #     from scipy.special import logsumexp

    #     gnb = GaussianNB()
    #     cnb = CategoricalNB()

    #     gnb_fit = gnb.fit(X_train[numericas], y_train)
    #     cnb_fit = cnb.fit(X_train[categoricas_ohe], y_train)

    #     log_joint_num = gnb._joint_log_likelihood(X_test[numericas])
    #     log_joint_cat = cnb._joint_log_likelihood(X_test[categoricas_ohe])

    #     class_log_prior = gnb.class_log_prior_

    #     combined_log_joint_likelihood = log_joint_num + log_joint_cat - class_log_prior

    #     log_posterior_proba = combined_log_joint_likelihood - logsumexp(combined_log_joint_likelihood, axis=1, keepdims=True)

    #     posterior_proba = np.exp(log_posterior_proba)

    #     predicted_classes = np.argmax(posterior_proba, axis=1)

    #     accuracy = accuracy_score(y_test, predicted_classes)

    #     return accuracy

    # _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    O algoritmo foi testado, mas devido à natureza mista (dados numéricos e categóricos) do dataset, é consideravelmente trabalhoso aplicar o algoritmo na base atual, mesmo com ajustes

    - https://stackoverflow.com/a/14255284
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Árvore de Decisão
    Estrutura que divide dados com regras, fácil de interpretar.
    """
    )
    return


@app.cell
def _(
    X_test,
    X_train,
    a,
    classification_reports,
    cm_heatmap,
    cr_table,
    y_test,
    y_train,
):
    from sklearn.tree import DecisionTreeClassifier
    # from sklearn.tree import export_graphviz
    # import graphviz


    def _():
        clf = DecisionTreeClassifier()

        clf = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cr = cr_table(y_test, y_pred)
        classification_reports["Árvore de Decisão"] = cr

        a(cr)
        a(cm_heatmap(y_test, y_pred))

        """
        dot_string = export_graphviz(
            clf,
            feature_names=numericas + list(categoricas_ohe),
            class_names=values,
        )

        graph = graphviz.Source(dot_string)

        a(type(graph))

        a(dir(graph))

        a(graph._repr_mimebundle_())

        a(graph)
        """


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Random Forest
    Conjunto de árvores que melhora precisão e reduz overfitting.
    """
    )
    return


@app.cell
def _(
    X_test,
    X_train,
    a,
    classification_reports,
    cm_heatmap,
    cr_table,
    y_test,
    y_train,
):
    from sklearn.ensemble import RandomForestClassifier


    def _():
        clf = RandomForestClassifier()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cr = cr_table(y_test, y_pred)
        classification_reports["Random Forest"] = cr

        a(cr)
        a(cm_heatmap(y_test, y_pred))


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### KNN
    Classifica com base na proximidade dos exemplos mais próximos.
    """
    )
    return


@app.cell
def _(
    X_test,
    X_train,
    a,
    classification_reports,
    cm_heatmap,
    cr_table,
    y_test,
    y_train,
):
    from sklearn.neighbors import KNeighborsClassifier


    def _():
        clf = KNeighborsClassifier()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cr = cr_table(y_test, y_pred)
        classification_reports["KNN"] = cr

        a(cr)
        a(cm_heatmap(y_test, y_pred))


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Support Vector Machines
    Separa classes com a maior margem possível entre elas.
    """
    )
    return


@app.cell
def _(
    X_test,
    X_train,
    a,
    classification_reports,
    cm_heatmap,
    cr_table,
    y_test,
    y_train,
):
    from sklearn.svm import SVC


    def _():
        clf = SVC()

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cr = cr_table(y_test, y_pred)
        classification_reports["Support Vector Machines"] = cr

        a(cr)
        a(cm_heatmap(y_test, y_pred))


    _()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Redes Neurais
    Modelo com várias camadas que aprende padrões complexos.
    """
    )
    return


@app.cell
def _(
    X_test,
    X_train,
    a,
    classification_reports,
    cm_heatmap,
    cr_table,
    y_test,
    y_train,
):
    from sklearn.neural_network import MLPClassifier


    def _():
        clf = MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(5, 2),
            random_state=1,
            max_iter=1000,
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        cr = cr_table(y_test, y_pred)
        classification_reports["Redes Neurais"] = cr

        a(cr)
        a(cm_heatmap(y_test, y_pred))


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""## Comparação""")
    return


@app.cell
def _(classification_reports):
    classification_reports
    return


@app.cell
def _(classification_reports):
    def map_cr_values(col, row):
        mapped = {}
        for k, v in classification_reports.items():
            mapped[k] = v[col][row]

        mapped = {
            k: v
            for k, v in sorted(
                mapped.items(), key=lambda item: item[1], reverse=True
            )
        }

        return mapped
    return (map_cr_values,)


@app.cell
def _(a, map_cr_values, mo, values):
    a(mo.md("Algoritmos mais balanceados (f1-score):"))


    def _():
        for k, v in enumerate(values):
            a(mo.md(v + ":"))
            mapped = map_cr_values(str(k), "f1-score")

            a(mapped)


    _()
    return


@app.cell
def _(a, map_cr_values, mo, values):
    a(
        mo.md("""
    Recall (sensibilidade): Dos casos que realmente são positivos, quantos o modelo conseguiu identificar corretamente. É a capacidade de "não perder" casos positivos.
    """)
    )


    def _():
        for k, v in enumerate(values):
            a(mo.md(v + ":"))
            mapped = map_cr_values(str(k), "recall")

            a(mapped)


    _()
    return


@app.cell
def _(a, map_cr_values, mo, values):
    a(
        mo.md("""
    Precision (Precisão): Das predições que o modelo disse serem positivas, quantas realmente eram positivas. É sobre "não errar quando diz que é positivo".
    """)
    )


    def _():
        for k, v in enumerate(values):
            a(mo.md(v + ":"))
            mapped = map_cr_values(str(k), "precision")

            a(mapped)


    _()
    return


@app.cell
def _(classification_reports, pd):
    dados = []
    for algoritmo, df_report in classification_reports.items():
        for (
            metrica
        ) in df_report.drop("support").index:  # 0, 1, 2, accuracy, macro avg, weighted avg
            for (
                classe
            ) in df_report.columns:  # precision, recall, f1-score, support
                valor = df_report.loc[metrica, classe]
                dados.append(
                    {
                        "algoritmo": algoritmo,
                        "classe": classe,
                        "metrica": metrica,
                        "valor": valor,
                    }
                )

    cr_melted = pd.DataFrame(dados)
    cr_melted
    return (cr_melted,)


@app.cell
def _(a, cr_melted, px):
    def _():    
        # Apenas classes 0, 1, 2 (sem accuracy, macro avg, weighted avg)
        classes_principais = ['0', '1', '2']
    
        for metrica in ['precision', 'recall', 'f1-score']:
            df_filtrado = cr_melted[
                (cr_melted['metrica'] == metrica) & 
                (cr_melted['classe'].isin(classes_principais))
            ]
        
            fig = px.bar(df_filtrado,
                         x='algoritmo',
                         y='valor',
                         color='classe', 
                         title=f'{metrica.capitalize()} - Classes 0, 1, 2',
                         barmode='stack')
        
            a(fig)

    _()
    return


@app.cell
def _(a, cr_melted, px):
    def _():
        df_pivot = cr_melted.pivot_table(
            index='algoritmo', 
            columns='metrica', 
            values='valor', 
            aggfunc='mean'  # Se houver múltiplas classes, faz a média
        )
    
        fig_heatmap = px.imshow(df_pivot,
                                title='Heatmap de Performance dos Algoritmos',
                                labels={'x': 'Métricas', 'y': 'Algoritmos', 'color': 'Valor'},
                                color_continuous_scale='RdYlBu_r')  # Vermelho=baixo, Azul=alto
    
        a(fig_heatmap)

    _()
    return


@app.cell
def _(a, cr_melted, mo, px):
    def _():
        a(mo.md("""
            O scatterplot é ideal para visualizar trade-offs entre métricas de machine learning. Ao plotar Precision vs Recall, por exemplo, você vê instantaneamente quais algoritmos sacrificam precisão para capturar mais casos positivos, ou vice-versa. Isso facilita enormemente a seleção do modelo ideal baseado no equilíbrio desejado para seu problema específico.
    Além dos trade-offs, o scatterplot revela correlações e outliers de forma visual. Algoritmos com comportamentos similares se agrupam, enquanto aqueles com performance incomum ficam isolados, ajudando na análise comparativa e na tomada de decisão sobre qual modelo usar em produção.
        """))
        # Criar DataFrame wide para scatterplot
        df_wide = cr_melted.pivot_table(
            index=['algoritmo', 'classe'], 
            columns='metrica', 
            values='valor'
        ).reset_index()
    
        # Matrix de scatter plots
        fig_scatter = px.scatter_matrix(df_wide,
                                       dimensions=['precision', 'recall', 'f1-score'],  # Excluir support por escala diferente
                                       color='algoritmo',
                                       title='Scatter Plot Matrix - Trade-offs entre Métricas')
    
        a(fig_scatter)

    _()
    return


if __name__ == "__main__":
    app.run()
