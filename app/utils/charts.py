import plotly.express as px

def bar_chart(df, x, y, title):
    return px.bar(df, x=x, y=y, title=title)