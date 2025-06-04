import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from scipy.sparse import csr_matrix
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model_knn, tfidf, scaler, X, df

    print("Modeller ve bileşenleri yükleniyor...")
    model_knn = joblib.load("model_files/knn_model.joblib")
    tfidf = joblib.load("model_files/tfidf_vectorizer.joblib")
    scaler = joblib.load("model_files/scaler.joblib")
    X = joblib.load("model_files/feature_matrix.joblib")
    df = pd.read_pickle("model_files/movies_df.pkl")

    print(f"Model yüklendi. {df.shape[0]} film içeren veri seti hazır.")

    yield

    print("API kapatılıyor, kaynaklar temizleniyor...")


app = FastAPI(
    title="Film Öneri API",
    description="K-NN algoritması kullanarak benzer film önerileri sunan bir API",
    version="1.0.0",
    contact={
        "name": "MocinEF Film Öneri Sistemi",
        "url": "https://mocinef.com",
    },
    lifespan=lifespan,
    openapi_tags=[
        {"name": "recommendations", "description": "Film önerisi işlemleri"},
        {"name": "movies", "description": "Film bilgisi işlemleri"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MovieRecommendationRequest(BaseModel):
    movie_ids: List[int] = Field(
        ...,
        example=[550, 155, 13],
        description="Öneriler için kullanılacak film ID'leri listesi",
    )
    top_n: Optional[int] = Field(
        5, example=5, description="Dönülecek öneri sayısı (maksimum 20)"
    )

    class Config:
        schema_extra = {
            "example": {
                "movie_ids": [
                    550,
                    155,
                    13,
                ],
                "top_n": 5,
            }
        }


class MovieResponse(BaseModel):
    id: int = Field(..., example=550, description="Film ID")
    title: str = Field(
        ..., example="The Shawshank Redemption", description="Film başlığı"
    )
    similarity: Optional[float] = Field(
        None, example=0.95, description="Benzerlik skoru (0-1 arası)"
    )
    poster_path: Optional[str] = Field(
        None, example="/shawshank-redemption.jpg", description="Film poster yolu"
    )
    vote_average: Optional[float] = Field(
        None, example=8.7, description="Ortalama puan (0-10 arası)"
    )
    release_date: Optional[str] = Field(
        None, example="1994-09-23", description="Yayın tarihi (YYYY-AA-GG formatında)"
    )
    genres: Optional[List[str]] = Field(
        None, example=["Drama", "Crime"], description="Film türleri"
    )


class MovieSearchResponse(BaseModel):
    id: int = Field(..., example=550, description="Film ID")
    release_date: Optional[str] = Field(
        None, example="1994-09-23", description="Yayın tarihi (YYYY-AA-GG formatında)"
    )
    title: str = Field(
        ..., example="The Shawshank Redemption", description="Film başlığı"
    )
    poster_path: Optional[str] = Field(
        None, example="/shawshank-redemption.jpg", description="Film poster yolu"
    )
    vote_average: float = Field(
        ..., example=8.7, description="Ortalama puan (0-10 arası)"
    )
    genres: Optional[List[str]] = Field(
        None, example=["Drama", "Crime"], description="Film türleri"
    )


@app.post(
    "/recommend",
    response_model=List[MovieResponse],
    summary="Film önerileri al",
    description="Verilen film ID'lerine göre benzer filmler önerir.",
    response_description="Benzerlik skoruna göre sıralanmış film önerileri",
    tags=["recommendations"],
)
def get_recommendations(request: MovieRecommendationRequest):
    movie_ids = request.movie_ids
    top_n = min(request.top_n, 20)

    if len(movie_ids) > 5:
        raise HTTPException(
            status_code=400, detail="En fazla 5 film ID'si gönderilebilir"
        )

    try:

        indices = []
        for movie_id in movie_ids:
            idx = df[df["id"] == movie_id].index
            if len(idx) == 0:
                raise HTTPException(
                    status_code=404, detail=f"ID: {movie_id} bulunamadı"
                )
            indices.append(idx[0])

        X_csr = X.tocsr()
        movie_vectors = [X_csr[idx] for idx in indices]

        combined_vec = sum(movie_vectors) / len(movie_vectors)

        distances, indices = model_knn.kneighbors(
            combined_vec, n_neighbors=top_n + len(movie_ids)
        )

        recommendations = []
        for j, i in enumerate(indices[0]):
            movie_id = int(df.iloc[i]["id"])

            if movie_id not in movie_ids:
                genres = []
                if isinstance(df.iloc[i].get("genres"), str) and df.iloc[i].get(
                    "genres"
                ):
                    genres = df.iloc[i].get("genres").split("-")
                elif isinstance(df.iloc[i].get("genres"), list):
                    genres = df.iloc[i].get("genres")
                recommendations.append(
                    {
                        "id": movie_id,
                        "title": df.iloc[i]["title"],
                        "similarity": round(float(1 - distances[0][j]), 3),
                        "poster_path": df.iloc[i]["poster_path"],
                        "vote_average": float(df.iloc[i]["vote_average"]),
                        "release_date": df.iloc[i]["release_date"],
                        "genres": genres,
                    }
                )
                if len(recommendations) >= top_n:
                    break

        return recommendations

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Öneri hesaplanırken hata oluştu: {str(e)}"
        )


@app.get(
    "/movies/search",
    response_model=List[MovieSearchResponse],
    summary="Film adına göre ara",
    description="Film adına göre arama yapar ve eşleşen filmleri döndürür. Boş arama terimi tüm filmleri döndürür.",
    response_description="Arama terimiyle eşleşen filmler listesi",
    tags=["movies"],
)
def search_movies(
    query: str = Query(
        None, description="Aranacak film adı (kısmi eşleşme, büyük/küçük harf duyarsız)"
    ),
    limit: int = Query(
        10, ge=1, le=100, description="Döndürülecek maksimum sonuç sayısı"
    ),
):
    if query is None or query.strip() == "":

        results = df.sort_values("popularity", ascending=False).head(limit)
    else:

        results = df[df["title"].str.contains(query, case=False, na=False)]

        results = results.head(limit)

    if len(results) == 0:
        return []

    response = []
    for _, movie in results.iterrows():
        genres = []
        if isinstance(movie.get("genres"), str) and movie.get("genres"):
            genres = movie.get("genres").split("-")
        elif isinstance(movie.get("genres"), list):
            genres = movie.get("genres")
        response.append(
            {
                "id": int(movie["id"]),
                "title": movie["title"],
                "poster_path": movie["poster_path"],
                "vote_average": float(movie["vote_average"]),
                "release_date": movie["release_date"],
                "genres": genres,
            }
        )

    return response


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Film Öneri API'ye hoş geldiniz! Dokümantasyon için /docs adresini ziyaret edin."
    }
