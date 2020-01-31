import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.

import html2text

##
st.title('Bike share')

city = st.text_input(label='Input location, e.g. Portland, Oregon, USA')


st.write('The current location is', city)


HtmlFile = open('portland_validation.html', 'r', encoding='utf-8')
html_code = HtmlFile.read() 

html_text = html2text.html2text(html_code)

# st.write(html_text)
st.markdown(html_text, unsafe_allow_html=True)