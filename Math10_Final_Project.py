import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pandas.api.types import is_numeric_dtype
pd.set_option('display.float_format', lambda x: '%.3f' % x)
rng = np.random.default_rng()

st.set_page_config(layout="wide")
st.title("Math10 Final Project")
st.markdown("[Kaihao Zhang](https://github.com/1879687161/Kaihao-Zhang)")
st.markdown("UCI ID: 21083348")
st.markdown("Please click my name to see my code!")

file = st.file_uploader("Upload file here:", type = ["csv"])

if file is not None:
    df = pd.read_csv(file)
    
    # Pandas part------------------------------------------------------------
    st.header("**_This is the Pandas Part:_**")
    df["date"] = pd.to_datetime(df["date"])
    st.subheader("Replace all empty entries, and delete the empty columns.")
    df = df.applymap(lambda x: np.nan if x ==" " else x )
    df.dropna(how='all', axis=1, inplace=True)
    df

    st.subheader("Get the dataframe that only include the numerical numbers.")
    numcols = [c for c in df.columns if is_numeric_dtype(df[c])]
    df2 = df[numcols]
    df2
    
    st.subheader("Check the highest new cases rows in LA")
    df_max = df.sort_values("new_cases",ascending = False)
    s = st.slider("Select you number", 0, 586, 10, 1)
    try1 = df_max.head(s)
    try1
    
    st.subheader("Check the state deaths that great than a celect number in LA")
    sd = st.slider("Select you number", 0, 75000, 10000, 100)    
    df_stdeath_1000 = df[df["state_deaths"]>sd]
    df_stdeath_1000
    
    st.subheader("Check the state remaining case that great than 1000 in LA")
    df_sd1000_nsc100 = df[(df["state_cases"]-df["state_deaths"])>1000]
    df_sd1000_nsc100
    
    #Get the new cases in CA that without LA
    df["new_case_noLA"] = df["new_state_cases"]-df["new_cases"]
    
    
    # Machine Learning part----------------------------------------
    st.header("**_This is the Machine Learning Part:_**")
    import sklearn
    import sklearn.linear_model
    assert sklearn.__version__ >= "0.20"
    
    st.markdown("Hint1: For example, you can choose the 'cases' as x, and 'deaths' as y. Then you can predict the possible number of deaths under the number of cases you selected.")
    st.markdown("Hint2: Make sure the two columns that you selected can make sense. Suggested selection: 'cases' as x, and 'deaths' as y; 'state_cases' as x, and 'state_deaths' as y; 'new_cases' as x, and 'new_deaths' as y")
    x = st.selectbox("Choose an x_value", numcols)
    y = st.selectbox("Choose an y_value", numcols)
    
    X = np.array(df[x]).reshape(-1,1)
    y = np.array(df[y]).reshape(-1,1)
    
    # Select a linear model
    model = sklearn.linear_model.LinearRegression()
    
    # Train the model
    model.fit(X, y)
    
    # Make a prediction
    mp = st.number_input("The number you want to predict:")
    Xpre = [[mp]] 
    try2 = model.predict(Xpre)
    st.subheader("Result:")
    try2
    
    st.markdown("Check the coefition of the model:")
    ch1 = model.coef_
    ch1
    st.markdown("Check the interception of the model:")
    ch2 = model.intercept_
    ch2
    
    st.markdown("**The cluster picture based on the Kmeans:**")
    from sklearn.cluster import KMeans
    kmeans = KMeans(6)
    kmeans.fit(df2)
    df2["cluster"] = kmeans.predict(df2)  
    al1 = alt.Chart(df2).mark_circle().encode(
        x = "new_state_cases",
        y = "new_state_deaths",
        shape = "cluster:N",
        color = "cluster:N"
    )
    al1 
    st.markdown("Based on the chart above it looks like I did not overfitting the data.")
    
    # Altair part
    st.header("**_This is the Altair Part:_**")
    st.write("Hint: You can panning and zooming the chart 1, 2, 4!! You can make a box and select the area that you like in Chart 3!")
    
    single_nearest = alt.selection_single(on='mouseover', nearest=True)
    selection = alt.selection_interval(bind='scales')
    brush = alt.selection_interval()
    
    st.markdown("This chart shows the relations between covid cases, and deaths in LA. We can clear see a pasitive tendence.")
    alt1 = alt.Chart(df).mark_circle().encode(
        x = "cases",
        y = "deaths",
        color = alt.Color("deaths", scale = alt.Scale(scheme = "goldred", reverse = False)),
        size = "deaths",
        tooltip = ["date", "cases", "deaths","state_cases", "state_deaths"]
    ).properties(
        width = 1280,
        height = 720,
        title = "LA_County_COVID_Cases"
    ).add_selection(
        selection
    )
    alt1

    st.markdown("This chart shows the relations between CA state covid cases, and deaths. We can clear see a pasitive tendence.")
    alt2 = alt.Chart(df).mark_circle().encode(
        x = "state_cases",
        y = "state_deaths",
        size = "state_deaths",
        color = alt.Color("state_deaths", scale = alt.Scale(scheme = "goldred",reverse=True)),
        tooltip = ["date", "state_cases", "state_deaths"]
    ).properties(
        width = 1280,
        height = 720,
        title = "California_COVID_Cases"
    ).add_selection(
        selection
    )
    alt2
    
    st.markdown("This chart shows the relations between new covid cases, and deaths in LA.")
    alt3 = alt.Chart(df).mark_bar().encode(
        x = "new_cases",
        y = "new_deaths",
        color = alt.condition(brush, alt.value('lightgray'), alt.Color("state_deaths", scale = alt.Scale(scheme = "turbo",reverse=True))),
        tooltip = ["date", "new_cases", "new_deaths"]
    ).properties(
        width = 1280,
        height = 720,
        title = "LA_New_COVID_Cases"
    ).add_selection(
        brush
    )
    alt3
    
    st.markdown("This chart shows the relations between CA state new covid cases, and deaths.")
    alt4 = alt.Chart(df).mark_circle().encode(
        x = "new_state_cases",
        y = "new_state_deaths",
        color = alt.Color("new_state_deaths", scale = alt.Scale(scheme = "turbo", reverse = False)),
        size = "new_state_deaths",
        tooltip = ["date", "new_state_cases", "new_state_deaths"]
    ).properties(
        width = 1280,
        height = 720,
        title = "California_New_COVID_Cases"
    ).add_selection(
        selection
    )
    alt4
    
    
    # Download data
    st.header("**_You can download the data that you selected hear!!!_**")
    sela = st.number_input("The row you want to start:", min_value=0, max_value=586, step=1)
    selb = st.number_input("The row you want to end:", min_value=0, max_value=586, step=1)
    row_val = df.loc[sela:selb]
    st.download_button(
        label = "Download selected rows as CSV",
        data = row_val.to_csv().encode('utf-8'),
        file_name = 'Selected_values.csv',
        mime='text/csv'
        )
    
    # Check Box
    st.header("**_This is the survey part, if you could leave your suggestion, that will be really helpful!_**")
    st.markdown("**Do you like my streamlit app?**")
    agree = st.checkbox("Yes, this app looks great!")
    disagree = st.checkbox("No, you still need to improve it!")
    if agree:
        st.write("Great! Thank you very much!")
        st.text_input("Please enter your suggestions hear:")
    elif disagree:
        st.write("Please write down your suggestions, thank you! I will improve myself!")
        st.text_input("Please enter your suggestions hear:")
    


    st.header("**_Reference Part_**")
    st.markdown("The link to the data that I used in this project: https://catalog.data.gov/dataset/la-county-covid-cases")
    st.markdown("Part of the code is taken from the course note: https://christopherdavisuci.github.io/UCI-Math-10/Proj/CourseProject.html")
    st.markdown("Part of the idea is taken from streamlit document: https://docs.streamlit.io/library/api-reference/media/st.image")
    st.markdown("Part of the idea is taken from this site: https://altair-viz.github.io/user_guide/interactions.html")
    
