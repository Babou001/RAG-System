# Updates
### 1. Multi page UI with streamlit 

**pages** directory that contains the following files : 
-  chatbot.py
-  document_mining.py
-  documents.py
-  home.py
  
Streamlit_app.py file to link all python files cited above.

### 2. unit tests 
- pytest librairie
- unit_test.py python file
 
### 3. users can view directly the pdf files from the UI instead of downloading it (document mining)
- The document_mining.py file from the pages directory has been modified to display PDF files in a popup, allowing the user to read the document by scrolling.


### multi user chatbot with a shared model instance between workers 
- An example with four workers and one instance of the model tested successfully with two workers.(fast_api_app.py)
- An example with two workers and model instances tested succesfully with two users. (app.py)