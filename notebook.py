import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

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
def _(a, aplicar_rotulos, categoricas, df, mo, rotulos):
    a(mo.md("##Gráficos exploratórios"))

    def _():
        rotulado = aplicar_rotulos(df, rotulos)

        for c in categoricas:
            a(rotulado[c].hist())

    _()
    return


if __name__ == "__main__":
    app.run()
