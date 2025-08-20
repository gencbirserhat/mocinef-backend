MOCINEF-BACKEND
===============

_Transforming Film Discovery with Intelligent Recommendations_

![last-commit](https://img.shields.io/github/last-commit/gencbirserhat/mocinef-backend?style=flat&logo=git&logoColor=white&color=0080ff) ![repo-top-language](https://img.shields.io/github/languages/top/gencbirserhat/mocinef-backend?style=flat&color=0080ff) ![repo-language-count](https://img.shields.io/github/languages/count/gencbirserhat/mocinef-backend?style=flat&color=0080ff)

_Built with the tools and technologies:_

![scikitlearn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=flat&logo=FastAPI&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white) ![Docker](https://img.shields.io/badge/Docker-2496ED.svg?style=flat&logo=Docker&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat&logo=SciPy&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic-E92063.svg?style=flat&logo=Pydantic&logoColor=white)

  

* * *

Table of Contents
-----------------

*   [Overview](#overview)
*   [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Usage](#usage)
    *   [Testing](#testing)

* * *

Overview
--------

mocinef-backend is a scalable API service built with FastAPI that delivers personalized movie recommendations and search functionalities powered by advanced machine learning models. It integrates various algorithms, including NMF, k-NN, BERT, and autoencoders, to provide accurate and relevant film suggestions. The project emphasizes containerized deployment with Docker, ensuring consistent environments and seamless scalability.

**Why mocinef-backend?**

This project simplifies deploying ML-driven movie recommendation systems. The core features include:

*   🟢 **FastAPI Service:** Efficiently handles recommendation and search requests with high performance.
*   🔵 **Multiple ML Models:** Leverages NMF, k-NN, BERT, and autoencoders for diverse recommendation strategies.
*   🟠 **Containerized Deployment:** Uses Docker for consistent, scalable, and portable environments.
*   🟣 **Robust Data Processing:** Incorporates feature scaling, TF-IDF vectorization, and pre-trained models for reliable results.
*   🟡 **Model Training Pipelines:** Supports scalable training workflows with GPU acceleration and notebooks.

* * *

Getting Started
---------------

### Prerequisites

This project requires the following dependencies:

*   **Programming Language:** JupyterNotebook
*   **Package Manager:** Pip
*   **Container Runtime:** Docker

### Installation

Build mocinef-backend from the source and install dependencies:

1.  **Clone the repository:**
    
        ❯ git clone https://github.com/gencbirserhat/mocinef-backend
        
    
2.  **Navigate to the project directory:**
    
        ❯ cd mocinef-backend
        
    
3.  **Install the dependencies:**
    

**Using [docker](https://www.docker.com/):**

    ❯ docker build -t gencbirserhat/mocinef-backend .
    

**Using [pip](https://pypi.org/project/pip/):**

    ❯ pip install -r requirements.txt
    

### Usage

Run the project with:

**Using [docker](https://www.docker.com/):**

    docker run -it {image_name}
    

**Using [pip](https://pypi.org/project/pip/):**

    python {entrypoint}
    

### Testing

Mocinef-backend uses the {**test\_framework**} test framework. Run the test suite with:

**Using [docker](https://www.docker.com/):**

    echo 'INSERT-TEST-COMMAND-HERE'
    

**Using [pip](https://pypi.org/project/pip/):**

    pytest
    

* * *

[⬆ Return](#top)

* * *
