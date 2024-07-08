import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csc_matrix
import streamlit as st
final_reting=pd.read_csv("final_reting.csv")
main_data=pd.read_csv("filtered_backup.csv")


book_pivot=final_reting.pivot_table(columns="user_id",index="title",values="rating")
book_pivot.fillna(0,inplace=True)
#model training

book_sparce=csc_matrix(book_pivot)
model=NearestNeighbors(algorithm="brute")
model.fit(book_sparce)


main_data.rename(columns={"Book-Title":"title","Book-Author":"auther","Year-Of-Publication":"year","Publisher":"publisher","Image-URL-L":"image"},inplace=True)
main_data=main_data[["title","auther","year","publisher","image","ISBN"]]


def predict_book(movie_name):
    movie_id = np.where(book_pivot.index == movie_name)[0][0]
    distances, indices = model.kneighbors(book_pivot.iloc[movie_id, :].values.reshape(1, -1), 6)
    titles = [book_pivot.index[i] for i in indices[0]]  # Collect titles
    return titles

st.title("Book Recommendation System")
st.write("select a book from the dropdown and click on the button to get the recommendations")
book_name=st.selectbox("Choose a book", book_pivot.index)
if st.button("Recommend"):
    if book_name in book_pivot.index:
        recommended_books=predict_book(book_name)
        for i in recommended_books:
            # st.write(
            #     main_data[main_data["title"]==i]["title"].values[0]
            # )
            
            col1, col2 = st.columns(2)  # Adjust the ratio as needed
            with col1:
                img=main_data[main_data["title"]==i]["image"].values[0]
                st.image(img, width=200) 
            with col2:
                name=main_data[main_data["title"]==i]["title"].values[0]
                isbn_no=main_data[main_data["title"]==i]["ISBN"].values[0]
                auther=main_data[main_data["title"]==i]["auther"].values[0]
                year=main_data[main_data["title"]==i]["year"].values[0]
                publisher=main_data[main_data["title"]==i]["publisher"].values[0]
                print(name,isbn_no,auther,year,publisher)
                st.header(name)
                st.write(f"ISBN: {isbn_no}")
                st.write(f"Auther: {auther}")
                st.write(f"Year: {year}")
                st.write(f"Publisher: {publisher}")


# At the end of your Streamlit app code (app.py), add the following:

footer="""<style>
.a {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color:#0e1117;
    color: white;
    text-align: center;
    text-decoration: none;
}
</style>
<div class="a">
  
  <p>&copy; 2024 Ramchandra All rights reserved.</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
