import streamlit as st
import langchain_helper

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine",("Indian","American","Korean"))

if cuisine:
    response = langchain_helper.generate_restaurant_name_and_menu(cuisine)
    st.header(response['restaurant_name'])
    menu_items = response['menu_items'].split(",")

    st.write("**Menu Items**")
    for item in menu_items:
        st.write("-",item)