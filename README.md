# IBM Machine Learning Professional Certificate
## Machine Learning Capstone Course

## Course-Recommender-App
A streamlit based Course Recommender System application.
This application was created as part of the final project for Machine Learning Capstone course for the IBM Machine Learning Professional Certificate.

The course recommender system designed as part of the Machine Learning Capstone project was used to create a Streamlit application. The application can be accessed here: [App Link](https://sheezer-course-recommender-app-recommender-app-6xy8bz.streamlit.app/)

The detailed work on the Machine Learning Capstone project can be accessed here: [Capstone Repo](https://github.com/sheezer/Data-Projects/tree/main/ML-Final%20Project)

### Notes:

1. To use the app successfully, go to the app link, wait for the app to boot and then select the model and tune the hyperparameters. Afterwards, check the courses and then press the Train Model button, followed by Recommend New Courses. The generated results will be at the end. Scroll down to see them.

2. For optimal results, check more courses from the provided catalog. For optimal results, check at least 10 courses.

3. Time taken for the app to run varies from model to model. 

4. The KNN model does not run because of space constraints on the streamlit platform. The code is fine but the required memory for KNN similarity matrix is very large. Therefore, **do not use the KNN model**.

### backend.py

The backend python file containing the code that runs in the background. Contains all the machine learning code.

### recommender_app.py

The recommender_app python file contains all the code required to create the streamlit app's basic structure.

### requirements.txt

The requirements file containing details of all the packages and their versions required to run the application successfully.

### course_embeddings.csv

Course embeddings data to be used in the app.

### course_genre.csv

Course genre data to be used in the app.

### course_processed.csv

Processed course data to be used in the app.

### courses_bow.csv

Bag-of-Words course data to be used in the app.

### ratings.csv

Ratings data to be used in the app.

### sim.csv

Similarity scores data to be used in the app.

### user_embeddings.csv

User embeddings data to be used in the app.

### user_profile.csv

User profile data to be used in the app.
