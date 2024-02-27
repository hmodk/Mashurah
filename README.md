## Abstract  
Mashurah is an innovative legal consultation chatbot that has been fed with a comprehensive database of 795 legal provisions across five fields: criminal law, mining investment systems, human resources, environmental law, and civil affairs law. It also supports both Arabic and English languages, making it an invaluable resource for a wide range of users seeking legal guidance.Mashurah is an innovative legal consultation chatbot that has been fed with a comprehensive database of 795 legal provisions across five fields: criminal law, mining investment systems, human resources, environmental law, and civil affairs law. It also supports both Arabic and English languages, making it an invaluable resource for a wide range of users seeking legal guidance.

## Introduction 
In light of the rapid development and the shift towards AI, more regulations and laws are being established. In response to this evolution, we conceived the idea of Mashurah, which aims to make legal consultations easier and faster. This is achieved by providing easy access to legal guidance and ensuring that everyone is promptly well-informed about their rights.

## Data Description and Structure
The data used in the project consists of 795 legal provisions and encompasses 15 legal systems across 5 different fields of law. This data was sourced from government websites, including the "Bureau of Experts" at the council of ministers. The data was available in various formats, with some being in tables, while the majority were in PDF and DOCX formats. Subsequently, each article underwent a preprocessing stage to convert the data into a single CSV dataset.

## Methodology
Mashurah was developed in Python and utilizes several libraries, including OS, Streamlit, Pandas, Langchain, and Dotenv. The project involves using two pre-trained models: a sentence transformer model for transferring data and user input and OpenAI's "GPT 3.5 Turbo" for processing user queries and generating responses. We used Streamlit to set up a user-friendly webpage for user interaction. Langchain was utilized for preprocessing and transferring the collected data to a CSV file that represents the used dataset.

## Conclusion and Future Work
In conclusion, Mashurah aims to ease the search and raise awareness among law enforcement practitioners and civilians about the context of 15 law systems across 5 different fields of law. The objective is to incorporate real cases from these fields, with a specific focus on criminal law, to provide users with legal guidance based on scenarios similar to their own, while also prioritizing the consideration of personal information privacy. The integration of real cases and a focus on privacy considerations underscores Mashurah's commitment to providing practical and ethical legal guidance, further enhancing its potential impact on legal awareness and education.

## Demo
